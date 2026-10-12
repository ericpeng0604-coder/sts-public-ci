#!/usr/bin/env python3
"""Arm G — 底层"会玩游戏"的统一模型:从零用 RL 训一个模型做【全部非战斗决策】
(选路 / 篝火 / 商店 / 事件 / 选卡)。战斗仍交 sts_lightspeed 的 MCTS。
统一打分器: f( 局面obs(412) ⊕ 候选描述符 ) -> 分;每个决策点对所有候选打分,softmax 选一。
候选描述符 = [决策类型one-hot] ⊕ [类型专属字段],无关字段填 0(详见下方 layout)。
reward = 整局最终楼层(稀疏延迟),REINFORCE + 移动平均 baseline。
跑(并行训练在 armG_train_parallel.py): ~/lab/physical-world-ai/.venv/bin/python armG_train.py
"""
import os, sys, json, re
SB = os.environ.get("STS_BOT_DIR", "../sts-bot")
sys.path.insert(0, os.path.join(SB, "sim/sts_lightspeed/build312"))
sys.path.insert(0, SB)
import slaythespire as sts
import torch, torch.nn as nn

# ---------- 卡牌词表(沿用 Arm S 的,稳定编码;并行下冻结只查不增)----------
VOCAB_PATH = os.path.join(SB, "armS_card_vocab.json")
VOCAB_CAP = 256
_vocab = json.load(open(VOCAB_PATH)) if os.path.exists(VOCAB_PATH) else {}
def card_name(c):
    m = re.search(r"Card (.+?)>", str(c));  return m.group(1) if m else str(c)
def card_idx(name):
    if name not in _vocab and len(_vocab) < VOCAB_CAP:
        _vocab[name] = len(_vocab)
    return _vocab.get(name, VOCAB_CAP - 1)
def save_vocab():
    json.dump(_vocab, open(VOCAB_PATH, "w"), ensure_ascii=False)

# ---------- 局面 obs 编码(归一化,沿用 Arm S)----------
_nn = sts.getNNInterface()
try:
    _maxes = [max(1.0, float(x)) for x in _nn.getObservationMaximums()]
except Exception:
    _maxes = None
def obs_vec(gc):
    o = list(_nn.getObservation(gc))
    if _maxes and len(_maxes) == len(o):
        o = [a / b for a, b in zip(o, _maxes)]
    return o                                          # 返回 list(并行下要 pickle)
OBS_DIM = len(_nn.getObservation(sts.GameContext(sts.CharacterClass.IRONCLAD, 1, 0)))

# ---------- 候选描述符 layout ----------
# 决策类型
DT_MAP, DT_REST, DT_SHOP, DT_EVENT, DT_CARD = 0, 1, 2, 3, 4
# 房间类型 -> 小索引(地图用)
ROOM_IDX = {sts.Room.MONSTER: 0, sts.Room.ELITE: 1, sts.Room.REST: 2,
            sts.Room.SHOP: 3, sts.Room.EVENT: 4, sts.Room.TREASURE: 5, sts.Room.BOSS: 6}
# 商店物品类型 -> 小索引
SITEM_IDX = {sts.RewardsActionType.CARD: 0, sts.RewardsActionType.RELIC: 1,
             sts.RewardsActionType.POTION: 2, sts.RewardsActionType.CARD_REMOVE: 3,
             sts.RewardsActionType.SKIP: 4}
EVENT_CAP = 64        # 事件 id one-hot 容量(够覆盖所有事件,越界 clamp)
EOPT_CAP = 8          # 事件选项序号 one-hot 容量

# 各段偏移
OFF_DTYPE = 0;                 W_DTYPE = 5
OFF_CARD  = OFF_DTYPE + W_DTYPE;  W_CARD = VOCAB_CAP        # 256:选卡 & 商店买卡共用
OFF_MROOM = OFF_CARD + W_CARD;    W_MROOM = 7              # 地图:目标房间
OFF_MLA1  = OFF_MROOM + W_MROOM;  W_MLA1 = 7               # 地图:下一行可达房间计数
OFF_MLA2  = OFF_MLA1 + W_MLA1;    W_MLA2 = 7               # 地图:再下一行可达房间计数
OFF_REST  = OFF_MLA2 + W_MLA2;    W_REST = 7               # 篝火:选项 one-hot
OFF_SITEM = OFF_REST + W_REST;    W_SITEM = 5              # 商店:物品类型
OFF_SPRICE= OFF_SITEM + W_SITEM;  W_SPRICE = 1             # 商店:价格(归一化)
OFF_EVID  = OFF_SPRICE + W_SPRICE; W_EVID = EVENT_CAP      # 事件:id one-hot
OFF_EOPT  = OFF_EVID + W_EVID;    W_EOPT = EOPT_CAP        # 事件:选项序号 one-hot
OFF_PASS  = OFF_EOPT + W_EOPT;    W_PASS = 1               # 通用:跳过/离开 标志
DESC_DIM  = OFF_PASS + W_PASS
INPUT_DIM = OBS_DIM + DESC_DIM

def _blank():
    return [0.0] * DESC_DIM

def _room_counts(gc, x, y, vec, off):
    """把 (x,y) 的子节点(第 y+1 行)按房间类型计数累加到 vec[off:off+7]。返回子节点列表。"""
    if y < 0 or y > 13:
        return []
    kids = gc.map_node_children(x, y)
    for cx in kids:
        ri = ROOM_IDX.get(gc.map_node_room(cx, y + 1))
        if ri is not None:
            vec[off + ri] += 1.0
    return kids

# ---------- 给当前画面的每个候选,产出 (描述符, 该候选对应的"执行闭包")----------
GOLD_MAX = 999.0
def build_choices(gc):
    """返回 (kind, [描述符...], [执行函数...])。执行函数接受 gc 并施加该选择。"""
    ss = gc.screen_state
    descs, execs = [], []

    if ss == sts.ScreenState.REWARDS:                     # 选卡(+skip)
        offered = gc.get_card_reward()
        if not offered:
            return ("card_empty", [], [])
        for c in offered:
            d = _blank(); d[OFF_DTYPE + DT_CARD] = 1.0; d[OFF_CARD + card_idx(card_name(c))] = 1.0
            descs.append(d); execs.append((lambda cc: (lambda g: g.pick_reward_card(cc)))(c))
        d = _blank(); d[OFF_DTYPE + DT_CARD] = 1.0; d[OFF_PASS] = 1.0      # skip
        descs.append(d); execs.append(lambda g: g.skip_reward_cards())
        return ("card", descs, execs)

    acts = sts.get_legal_game_actions(gc)
    if not acts:
        return (str(ss), [], [])

    if ss == sts.ScreenState.MAP_SCREEN:
        cy = gc.cur_map_node_y
        for a in acts:
            dx, dy = a.idx1, cy + 1
            d = _blank(); d[OFF_DTYPE + DT_MAP] = 1.0
            ri = ROOM_IDX.get(gc.map_node_room(dx, dy))
            if ri is not None: d[OFF_MROOM + ri] = 1.0
            elif dy >= 15: d[OFF_MROOM + ROOM_IDX[sts.Room.BOSS]] = 1.0
            kids = _room_counts(gc, dx, dy, d, OFF_MLA1)        # 下一行
            for cx in kids:                                     # 再下一行
                _room_counts(gc, cx, dy + 1, d, OFF_MLA2)
            descs.append(d)
        execs = [(lambda aa: (lambda g: aa.execute(g)))(a) for a in acts]
        return ("map", descs, execs)

    if ss == sts.ScreenState.REST_ROOM:
        for a in acts:
            d = _blank(); d[OFF_DTYPE + DT_REST] = 1.0
            if 0 <= a.idx1 < W_REST: d[OFF_REST + a.idx1] = 1.0
            descs.append(d)
        execs = [(lambda aa: (lambda g: aa.execute(g)))(a) for a in acts]
        return ("rest", descs, execs)

    if ss == sts.ScreenState.SHOP_ROOM:
        shop_cards = gc.get_shop_cards()
        for a in acts:
            d = _blank(); d[OFF_DTYPE + DT_SHOP] = 1.0
            t = a.rewards_action_type
            if t in SITEM_IDX: d[OFF_SITEM + SITEM_IDX[t]] = 1.0
            if t == sts.RewardsActionType.SKIP: d[OFF_PASS] = 1.0
            if t == sts.RewardsActionType.CARD and 0 <= a.idx1 < len(shop_cards):
                card, price = shop_cards[a.idx1]
                d[OFF_CARD + card_idx(card_name(card))] = 1.0
                d[OFF_SPRICE] = min(price, GOLD_MAX) / GOLD_MAX if price and price > 0 else 0.0
            descs.append(d)
        execs = [(lambda aa: (lambda g: aa.execute(g)))(a) for a in acts]
        return ("shop", descs, execs)

    if ss == sts.ScreenState.EVENT_SCREEN:
        ev = max(0, min(EVENT_CAP - 1, gc.cur_event))
        for a in acts:
            d = _blank(); d[OFF_DTYPE + DT_EVENT] = 1.0
            d[OFF_EVID + ev] = 1.0
            if 0 <= a.idx1 < EOPT_CAP: d[OFF_EOPT + a.idx1] = 1.0
            descs.append(d)
        execs = [(lambda aa: (lambda g: aa.execute(g)))(a) for a in acts]
        return ("event", descs, execs)

    return (str(ss), [], [])    # 不该到这(没开对应钩子)

# ---------- 打分网络:共享 MLP f(obs ⊕ 描述符) -> 分 ----------
class Scorer(nn.Module):
    def __init__(self, arch=(128, 128)):
        super().__init__()
        layers, prev = [], INPUT_DIM
        for h in arch:
            layers += [nn.Linear(prev, h), nn.ReLU()]; prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
    def score(self, o, descs):
        """o: tensor[OBS_DIM]; descs: list[list[DESC_DIM]] -> scores tensor[k]"""
        rows = torch.stack([torch.cat([o, torch.tensor(d, dtype=torch.float32)]) for d in descs])
        return self.net(rows).squeeze(-1)

SIMCOUNT = int(os.environ.get("STS_SIM_COUNT", "2000"))
ASC = int(os.environ.get("ASC", "0"))       # Ascension 难度(训练/评测同一难度=分布匹配)

def play_game(seed, net, train=True, max_steps=600):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, ASC)
    ag = sts.Agent(); ag.simulation_count_base = SIMCOUNT
    ag.pause_on_card_reward = ag.pause_on_map = ag.pause_on_rest = ag.pause_on_shop = ag.pause_on_event = True
    traj = []; steps = 0
    while gc.outcome == sts.GameOutcome.UNDECIDED and steps < max_steps:
        steps += 1
        ag.playout(gc)
        if gc.outcome != sts.GameOutcome.UNDECIDED:
            break
        kind, descs, execs = build_choices(gc)
        if not descs:                                  # 空候选(如选卡空):放过
            if gc.screen_state == sts.ScreenState.REWARDS:
                gc.skip_reward_cards()
            else:
                break                                  # 不该发生
            continue
        if len(descs) == 1:                            # 单候选=无真决策,直接执行不记梯度
            execs[0](gc); continue
        o = obs_vec(gc)
        with torch.no_grad():                          # rollout 只为选动作,不留梯度(主进程重算)
            scores = net.score(torch.tensor(o, dtype=torch.float32), descs)
            probs = torch.softmax(scores, dim=0)
            if train:
                a = torch.multinomial(probs, 1).item()
                traj.append((o, descs, a))             # 纯数据,可 pickle
            else:
                a = int(torch.argmax(probs).item())
        execs[a](gc)
    win = gc.outcome == sts.GameOutcome.PLAYER_VICTORY
    return {"seed": seed, "floor": gc.floor_num, "act": gc.act, "win": win,
            "hp": gc.cur_hp, "deck": len(gc.deck), "traj": traj}

def read_seeds(fn):
    return [int(x) for x in open(os.path.join(SB, fn)) if x.strip() and not x.startswith("#")]

if __name__ == "__main__":
    print(f"INPUT_DIM={INPUT_DIM} = obs {OBS_DIM} ⊕ desc {DESC_DIM}", flush=True)
    print(f"desc layout: dtype@{OFF_DTYPE} card@{OFF_CARD} mroom@{OFF_MROOM} mla1@{OFF_MLA1} "
          f"mla2@{OFF_MLA2} rest@{OFF_REST} sitem@{OFF_SITEM} sprice@{OFF_SPRICE} "
          f"evid@{OFF_EVID} eopt@{OFF_EOPT} pass@{OFF_PASS}", flush=True)
