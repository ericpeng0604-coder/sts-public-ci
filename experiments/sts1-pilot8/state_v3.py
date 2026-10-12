"""Public noncombat decision-time state; unavailable values stay None."""
import pathlib,json,re,hashlib,collections
from catalog import IDS,NAMES,RELIC_IDS

FIELDS=['hp','max_hp','gold','relics','potions','current_act_boss','visible_route_choices','event_name']

def native_card(c):
    cid=int(c.id)
    if not 0<cid<len(IDS):raise ValueError('native_card_unknown')
    n=int(c.upgrade_count)
    return IDS[cid]+('+'+str(n) if n else '')

def native_state(gc,policy,kind,descs,ascension):
    x,y=int(gc.cur_map_node_x),int(gc.cur_map_node_y)
    routes=[]
    if 0<=y<14:
        routes=[{'x':int(cx),'y':y+1,'room':str(gc.map_node_room(cx,y+1)).split('.')[-1]} for cx in gc.map_node_children(x,y)]
    state={'schema':'sts1-noncombat-public-state-v3','kind':kind,'floor':int(gc.floor_num),'act':int(gc.act),
           'ascension':ascension,'character':'IRONCLAD','deck':[native_card(c) for c in gc.deck],
           'hp':int(gc.cur_hp),'max_hp':int(gc.max_hp),'gold':int(gc.gold),
           'relics':[RELIC_IDS[int(r.id)] for r in gc.relics], 'potions':list(gc.potions),
           'current_act_boss':str(gc.boss).split('.')[-1],
           'visible_route_choices':routes,
           'event_name':str(gc.cur_event).split('.')[-1] if kind=='event' else None,
           'provenance':{'source':'native_decision_before_execute'}}
    state['known']={k:state.get(k) is not None for k in FIELDS}
    state['status']={k:'known' if state.get(k) is not None else 'not_applicable' if k=='event_name' and kind!='event' else 'unknown' for k in FIELDS}
    return state

def human_state(row,run):
    floor=row['floor'];state={'schema':'sts1-noncombat-public-state-v3','kind':'card','floor':floor,
        'act':1 if floor<=16 else 2 if floor<=33 else 3 if floor<=51 else 4,
        'ascension':run.get('ascension_level'),'character':'IRONCLAD','deck':row['deck']}
    state.update({k:None for k in FIELDS})
    # A chronological relic list can be reconstructed only before any event,
    # with no same-floor gains; post-floor HP/gold and future boss logs are never used.
    if not any(int(e['floor'])<=floor for e in run.get('event_choices',[])) and not any(int(e['floor'])==floor for e in run.get('relics_obtained',[])):
        starter=[] if run.get('neow_bonus')=='BOSS_RELIC' else ['Burning Blood']
        log=run.get('neow_bonus_log')
        if isinstance(log,dict):
            state['relics']=starter+list(log.get('relicsObtained',[]))+[e['key'] for e in run.get('relics_obtained',[]) if int(e['floor'])<floor]
    state['known']={k:state.get(k) is not None for k in FIELDS}
    state['status']={k:'known' if state.get(k) is not None else 'not_applicable' if k=='event_name' else 'unknown' for k in FIELDS}
    state['provenance']={'source':'human_history','unavailable_reason':'No exact before-choice capture; per-floor summaries cannot establish timing.'}
    return state

def action_names(gc,policy,kind,descs):
    if kind=='card':
        actions=[native_card(c) for c in gc.get_card_reward()]+['SKIP']
    else:
        semantics=[policy.describe_choice(kind,d) for d in descs]
        if kind=='map':
            x,y=int(gc.cur_map_node_x),int(gc.cur_map_node_y)
            children=list(gc.map_node_children(x,y)) if 0<=y<14 else []
            if len(children)==len(descs):
                semantics=[{**s,'visible_target_x':int(cx),'visible_target_y':y+1} for s,cx in zip(semantics,children)]
        actions=[kind.upper()+':'+json.dumps(s,sort_keys=True,separators=(',',':')) for s in semantics]
    if len(actions)!=len(descs) or len(set(actions))!=len(actions):raise ValueError('action_semantics_ambiguous:'+kind)
    return actions

def features(state,action):
    # Explicit missingness: unknown health is not health=0.
    stem=action.split('+')[0];kind=state['kind'];deck=collections.Counter(c.split('+')[0] for c in state['deck'])
    f={'bias:'+stem:1.0,'kind:'+kind:1.0,'floor:'+stem:state['floor']/57,'deck_size:'+stem:len(state['deck'])/50,
       'copies:'+stem:min(deck.get(stem,0),4)/4,'asc:'+stem:state['ascension']/20,
       'upgraded:'+stem:float('+' in action)}
    for card,n in deck.items():f['pair:'+stem+':'+card]=min(n,4)/4
    for field in FIELDS:f['known:'+stem+':'+field]=float(state.get(field) is not None)
    if state.get('hp') is not None and state.get('max_hp'):
        f['hp_ratio:'+stem]=state['hp']/state['max_hp']
    if state.get('gold') is not None:f['gold:'+stem]=min(state['gold'],999)/999
    for field in ['relics','potions']:
        if state.get(field) is not None:
            for item in state[field]:f[field+':'+stem+':'+str(item)]=1.0
    if state.get('current_act_boss') is not None:f['boss:'+stem+':'+state['current_act_boss']]=1.0
    if state.get('visible_route_choices') is not None:
        rooms=collections.Counter(r.get('room','unknown') for r in state['visible_route_choices'])
        for room,n in rooms.items():f['route:'+stem+':'+room]=n/4
    if state.get('event_name') is not None:f['event:'+stem+':'+state['event_name']]=1.0
    return f
