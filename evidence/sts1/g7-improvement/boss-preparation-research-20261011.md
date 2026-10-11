# G7 下一步研究：Boss 前牌組與資源準備

研究日期：2026-10-11。範圍：Issue #19；研究文件，不是候選預註冊或實驗結果。
基底：`sts-public-ci` PR #34 最新 head `59e091c6e98e16d3197fcfd78baffff1b5a82b39`。

## 結論與建議

優先建立 **Boss 前準備診斷 → 一個有證據支持的非戰鬥決策改動 → 整局配對評估**。
現成 AI 可借用的共同方法是：把當前牌組對特定敵人的能力與健康代價，接回選牌、篝火和路線決策。
最接近的是 spire-agent 的 encounter readiness／needs；最容易做透明基線的是 Bottled AI 的篝火 handler；MilesOram 提供戰鬥與宏觀規劃之間的介面；AscensionAI 提醒必須核對模型的觀察與控制範圍。

本專案尚未證實「Boss 前準備不足」是可修正的主要死因。建議先在新的 Train-only 資料中區分入場血量、牌組能力、藥水與戰鬥策略的問題，再決定下一個機制。
第一個介入優先考慮最後篝火的升級選擇；只有資料支持休息／升級取捨錯誤，才改這個取捨。
若問題在沒有成長或抽不到關鍵牌，應改一次 reward pick/skip，而不是一律多休息。
這是研究推論，勝率提升 **NOT_VERIFIED**。G7 保留。

## 1. 已確認的本專案證據

- H22：Train10、Probe10、Dev30，雙方合計 9/50 勝；每階段 C-only/P-only 都是 0/0。19/30 Dev 種子有效改變決策，仍沒有新勝場。50 組是探索階段的描述性合計，不能當正式確認或等效性證明。[H22 報告](./round-016-h22-paired-report.md)
- Round014 的 10 場新 Train 診斷全部敗北：7 場終止於 floor50、2 場 floor33、1 場 floor47。6 場最後戰鬥以 >80% HP 開始，其餘 4 場 <60%。所以低入場血量不能解釋全部敗局；這個小樣本也不能代表完整 G7 死因分布。[GitHub 原始診斷報告](https://github.com/ericpeng0604-coder/sts-public-ci/blob/59e091c6e98e16d3197fcfd78baffff1b5a82b39/evidence/sts1/g7-improvement/round-014-live-train-report.md)
- 舊 run-context 藥水觀察曾在戰鬥內過時。修正後需用 BattleContext 的 live inventory；「記錄有藥水」不等於當時真的能使用，更不等於能救活。[資源審計](./train-resource-audit-20261010.md)
- 下一輪不能挖 H22 Probe/Dev 決策來產生策略；既有未使用 reserved pools 仍排除。以新的、互不重疊的 Train 診斷與候選驗證池開始。[Issue #19](https://github.com/ericpeng0604-coder/sts-public-ci/issues/19)

以上是已發表證據重用；本次沒有打開大型 Episode、原始軌跡或 Formal outcome，也沒有新增遊戲。

## 2. 四個現成方案怎麼做

### A. spire-agent：把「能不能打」接到「怎麼準備」

核對版本：`8fcaa767479f62aa2528d2e9be260de97172c6b0`；MIT。

- [readiness.py](https://github.com/AttemorySystem/spire-agent/blob/8fcaa767479f62aa2528d2e9be260de97172c6b0/src/spire_agent/tools/map/readiness.py)：以當前 HP、deck、relic 等建立獨立 encounter 模擬，彙整勝局、樣本量、Wilson 區間、存活率與勝局剩餘 HP；依 fingerprint 快取。區分 `SUPPORTED`、`AT_RISK`、`INCONCLUSIVE`。作者預設 worlds=8、simulations=100、minimum_survival=0.60，這些不是本專案已驗證門檻。
- [needs.py](https://github.com/AttemorySystem/spire-agent/blob/8fcaa767479f62aa2528d2e9be260de97172c6b0/src/spire_agent/tools/winning_path/needs.py)：`plan_targets()` 先由 state 定義 Boss／Act 目標，避免看到 offered cards 才改需求；`analyze_need_profile()` 比較敵人 requirements 與牌組 capabilities，並考慮來源密度。缺 Boss 身分就以 Act pool 評估並標缺失；未驗證 capability 不直接算成立。
- [run_keys.py](https://github.com/AttemorySystem/spire-agent/blob/8fcaa767479f62aa2528d2e9be260de97172c6b0/src/spire_agent/tools/run_keys.py)：`rest_policy()` 比較目前 route readiness 和休息後 readiness，核對 entry HP，處理必要生存與 key deadline，不只用固定 HP 門檻。
- [potion_gate.py](https://github.com/AttemorySystem/spire-agent/blob/8fcaa767479f62aa2528d2e9be260de97172c6b0/src/spire_agent/tools/mcts/potion_gate.py)：先評估不用藥，再在 danger/emergency 試少量藥水；需有實質收益才釋放。unknown 不被當作安全或可任意開放。

原碼限制很重要：`readiness.py` 的預設 BOSS group 只在 Act1/2 且 Boss 已知時產生 target，Act3 group 為空；`_spec()` 將 potion slots 全部設成空槽，不能代表真實藥水準備度；它要求並傳入實際 seed、act、floor。不能直接宣稱這套預設已涵蓋本專案主要的 Act3 Boss 死亡，或符合本研究獨立 probe RNG／不讀未來的要求。
其固定 submodule 的 [ProspectiveBattle.cpp](https://github.com/Attemory/sts_lightspeed/blob/46e14e4adc23d2c1c738df8902a6f297109fee50/src/convert/ProspectiveBattle.cpp) 是以 seed/floor/world 重建 synthetic encounter，不是還原原 run 未來 Boss。傳 seed 本身不證明偷看未來；需檢查 world 生成與可見資訊界線，才能判定本地評估合規與忠實度。
可借用：敵人需求、缺口分類、rest 前後比較、unknown handling、診斷理由與快取。
不建議本輪接整套 LLM 或直接搬作者門檻。readiness 的生成戰鬥可能不同於真實後續 Boss，必須分開驗證狀態語意、RNG 與模擬忠實度。

成績界線：[README](https://github.com/AttemorySystem/spire-agent/blob/8fcaa767479f62aa2528d2e9be260de97172c6b0/README.md) 宣稱多次 A20H wins；近期 Ironclad 12-run 樣本平均 floor38.6、6 次到 Act3、2 次到 Act4，不能重算為與 G7 同條件的配對勝率。Linux/macOS tested，Windows 明確 untested。本機相容與效果 **NOT_VERIFIED**。

### B. Bottled AI：可直接審查的策略 handler

核對版本：`7b47a877d626cdb4fbeeb423c46472b15dd2ac00`；MIT。

- [common_campfire_handler.py](https://github.com/xaved88/bottled_ai/blob/7b47a877d626cdb4fbeeb423c46472b15dd2ac00/rs/common/handlers/common_campfire_handler.py)：一般篝火 HP≤60% 休息；floor49 採 HP≤85%；Pantograph 修正 Boss 前恢復需求，再安排去詛咒、重要升級等。這是特定框架的規則，不是最優健康線。
- [Ironclad potions_handler.py](https://github.com/xaved88/bottled_ai/blob/7b47a877d626cdb4fbeeb423c46472b15dd2ac00/rs/ai/requested_strike/handlers/potions_handler.py)：Boss turn1 使用未排除的可用藥水；Elite 有 HP 條件，另有特定敵人的 target 處理。
- [shop_purchase_handler.py](https://github.com/xaved88/bottled_ai/blob/7b47a877d626cdb4fbeeb423c46472b15dd2ac00/rs/ai/requested_strike/handlers/shop_purchase_handler.py)：先除 curse、購買 Perfected Strike 等，反映 RequestedStrike 特定構築，不能通用移植。

可借用：分開一般／Boss 前最後篝火，納入遺物恢復效果，做獨立升級或藥水 ablation。
避免同時搬商店、路線、選牌、藥水四套規則。藥水「Boss 一開始全用」也不能當成經驗定理。

成績界線：[winrates.md](https://github.com/xaved88/bottled_ai/blob/7b47a877d626cdb4fbeeb423c46472b15dd2ac00/docs/winrates.md) 的 2024-10 release03：Ironclad 20%／50 場；52% 是 Watcher。作者也提醒樣本不夠高信心。非配對，難度／Heart 與 G7 一致性 **NOT_VERIFIED**，20% 與18%不能證明差異。README 有 Windows 設定說明；本地安裝／跑通 **NOT_RUN**。

### C. MilesOram/STS：用戰鬥健康代價評估宏觀選項

核對版本：`3dec2528d3fbf9ad493f987b6947d6543a1353a1`；GitHub 未標明 license，僅借方法，不直接複製原碼。

- [README](https://github.com/MilesOram/STS/blob/3dec2528d3fbf9ad493f987b6947d6543a1353a1/README.md) 與 [作者詳解](https://github.com/MilesOram/STS/blob/3dec2528d3fbf9ad493f987b6947d6543a1353a1/WriteUp/In-Depth%20Write-Up.md)：DQN 打戰鬥；第一 NN 預測目前 deck/potions/relics 對各 encounter 的 HP loss 與誤差；第二 NN 預測 terminal floor，模擬非戰鬥選項後選較高者。RNG 結果用樣本平均。
- [PRED.py](https://github.com/MilesOram/STS/blob/3dec2528d3fbf9ad493f987b6947d6543a1353a1/RL/PRED.py)：126 outputs，更新對應 fight 的損失與 absolute residual proxy。不能把作者稱的 variance 寫成已校準統計變異數。
- [NN.py](https://github.com/MilesOram/STS/blob/3dec2528d3fbf9ad493f987b6947d6543a1353a1/RL/NN.py)：`callExpectedforSim()` 把健康代價估計接到 macro state；目標是 terminal floor，並非直接最大化整局勝率。

可借用：讓「牌組打這個敵人的代價」成為篝火／路線決策輸入。先驗證本專案模擬器的只讀 clone 能力，再考慮估計器；不需要立刻重訓兩個模型。
作者也承認 micro/macro 分離會漏掉 gold、maxHP、relic counters 對下一段路的價值，且 potion 學習有限。

成績界線：自建 Ironclad C++ replica，Ascension 未實作；4% 是訓練期間最佳連續100局，作者明言未評估固定模型。不能視為 frozen benchmark，也不能與 G7 比優劣。Windows 移植 **NOT_VERIFIED**。

### D. AscensionAI：先確認模型看得到牌組、真的能改牌組

核對版本：`b34a2cd80088222be0a46285aee8e22aebf2ef55`；MIT。

- [obs_encoder.py](https://github.com/JustinoChan/AscensionAI/blob/b34a2cd80088222be0a46285aee8e22aebf2ef55/scripts/obs_encoder.py)：原有 deck aggregate 之外，追加 per-card deck count vector，保留原特徵位置以 warm transfer；map 有 rest/elite/combat density 與剩餘樓層。
- [screen_handler.py](https://github.com/JustinoChan/AscensionAI/blob/b34a2cd80088222be0a46285aee8e22aebf2ef55/scripts/screen_handler.py)：非 BC 模式把升級／移除選擇交給 policy；BC 仍用 heuristic 示範。
- [sts_gym_env.py](https://github.com/JustinoChan/AscensionAI/blob/b34a2cd80088222be0a46285aee8e22aebf2ef55/scripts/sts_gym_env.py)：以平均 card quality 和 upgrade bonus 建 deck potential，獎勵其變化。這種代理分数容易偏離具體 Boss 與完整勝場，不能拿 deck score 成長當胜率。

可借用：檢查 observation 與 action ownership，追查 upgrade/removal 是否被 handler 截走；後續才決定是否需要新模型輸入。
本 Issue 禁止 PPO 變更，因此不移植其訓練方案。

成績界線：[README](https://github.com/JustinoChan/AscensionAI/blob/b34a2cd80088222be0a46285aee8e22aebf2ef55/README.md) 仍有585維舊說明和舊評估，不能據此斷言 deck vector 未實作。38.1% 是 Boss encounter 勝率，不是整局勝率；文件明言尚無完整勝利。新增 deck-building 版本的 frozen paired win gain **NOT_VERIFIED**。Windows 有文件化使用路徑，本地驗收 **NOT_RUN**。

## 3. 本專案能接什麼，還缺什麼

在基底 [simulator.py](https://github.com/ericpeng0604-coder/sts-public-ci/blob/59e091c6e98e16d3197fcfd78baffff1b5a82b39/src/roguelike_ai/sts1_phase3/simulator.py) 已確認：

- 實際 G7 非戰鬥路徑是 `ArmGNoncombatPolicy`：載入 frozen `armG_train`、score 每個 candidate，再選擇；不是 `deterministic_noncombat_step()` 的 first-card/first-legal fallback。
- `training_vector_snapshot()` 記錄412維 observation 和368維 candidate descriptors，但向量長度本身不證明已含 Boss-aware needs、完整牌組語意或所有升級子選擇。外部 compatibility root 的實際特徵和 action coverage 需另核對，**NOT_VERIFIED**。
- `deck_snapshot()` 可記錄牌名與升級；`_diagnostic_run_snapshot()` 有 HP、牌組、遺物、run potions；`_combat_potion_inventory_snapshot()` 分開 live battle inventory。
- `describe_choice()` 能區分 card/skip、map/lookahead、shop/type/price、rest option；既有診斷可重用，不能因此假定已有 Boss readiness evaluator。

最低所需新資料：決策當時可見 Boss 身分、精確 deck/upgrades、遺物狀態、當前資源、剩餘合法路線、候選動作與子選擇、G7 原決策及理由、state/prefix fingerprint。
Boss 若當時不可見，只能使用可見候選 pool 並標 unknown；禁止讀未來 RNG 或從最後遇到的 Boss 回填決策時特徵。
所有 state-copy、預覽 upgrade、fork combat 及 RNG 隔離 API 的可用性仍 **NOT_VERIFIED**。

## 4. 建議研究設計

### 第一步：新 Train-only 準備診斷

先對新且排除既有／reserved pools 的10個種子，跑 unchanged G7；預先登記第一場 trace-off replay。共11整局，完整保留原始證據。
本次報告沒有分配或消耗這些種子。

在每 Act 開始、card reward、shop、campfire、Boss 入場記錄：

| 欄位 | 要回答的問題 | 注意事項 |
| --- | --- | --- |
| Boss／可見候選 pool | 是為哪個敵人準備？ | 不從終局反推可見資訊 |
| HP、恢復遺物、剩餘路線 | 是否因途中消耗太多才進場不足？ | 看整段路，不只最後 HP |
| 爆發／成長／抽牌／防禦來源 | 能否在這個 matchup 結束戰鬥、度過壞抽？ | 來源數是代理指標，不是已證明能力 |
| 牌組大小、費用、升級、curse/status | 回合能否打出關鍵牌？ | 不把同名牌粗暴算成同等價值 |
| 藥水、空位、合法使用動作 | 缺資源、過早消耗，還是沒有效用？ | 戰鬥用 live inventory |
| 原候選動作與 afterstate | 哪個具體非戰鬥選擇可能修正缺口？ | 不以死局标签直接決定新規則 |

先分類：血量／耗損、長戰能力、抽牌與費用、特定 Boss matchup、藥水、戰鬥 planner／未知。
若沒有跨種子可重現的決策錯誤，停止該介入，不為了產生新版本硬改。

### 第二步：只讀 readiness 與機制反事實

先做 capability checklist；可靠 clone API 可用時，才加 encounter probes。
評估目前狀態與候選升級／休息的差異，記錄勝敗、HP loss 分布、样本量與不確定性；若使用篩選過的勝局剩餘 HP，必須標 conditional-on-win。

- 使用相同 paired worlds／combat budget，比較具體 legal action；不得偷用實際將來的 shuffle 或 seed。
- 優先核對 Act3 Boss，並分開無藥／真實藥水庫存的條件；不能把 spire-agent 的空 Act3 group 或空藥水 probe 當成完整資源驗收。
- 不更改真實 RNG、G7 或原狀態；trace、probe、cache 開關都需 invariance 檢查。
- 快取 identity 至少包含完整準備狀態、encounter/難度、planner/模擬器版本、預算與 probe RNG scheme；資源更新後必須失效。
- 生成 Boss battle 的勝率只是 proxy。單個 Boss-start rescue 或 readiness score 上升不能算 full-run gain；必須接回完整整局配對。
- 中途虛擬戰鬥要另外預註冊 compute/counterfactual 預算；若 fork 跑成完整 Episode，計入 Issue 的整局額度。API、成本未確認前不假定免費或無限可試。

### 第三步：選一個機制，凍結後才做配對

優先順序為研究安排，不是已選好的 H23：

1. **最後篝火升級目標**：若多個 Train cases 證明 G7 升級了無法補 matchup 缺口的牌，僅更換 smith 的具體目標；先確認這個選擇可被獨立控制。
2. **最後篝火 rest/upgrade**：只有低生存樣本及反事實支持時才測；不得給所有 Boss 套固定85%休息線，且必須納入遺物及特定 Boss 機制。
3. **一個 reward pick/skip**：若問題在成長或一致性缺口，測一個 Boss-aware 選牌條件；避免硬套 Perfected Strike／通用 card tier list。

藥水保留／購買與路線調整排後：它們涉及多場機會成本，需要更強的時序證據；不能無條件省藥至 Boss 而在途中死亡。
最多研究三種假說，但本建議執行輪只驗證一個主要改動，避免混合因果。

候選規則的適用 Boss、牌／遺物清單、score、coverage、unknown fallback、budget 和停止條件，須由 Train 證據支持並在新候選 episodes 前寫成獨立預註冊。
缺牌語意、升級效果、Boss 身分或精確 legal action 時保持 G7；不猜新動作。

## 5. 評估與驗收順序

建議上限：診斷11整局 + 新 Train10 配對21整局（含一次預註冊 invariance replay）+ Probe10 20整局 + Dev30 60整局 = **112整局**。
這是尚未執行的預算草案，低於每普通探索輪140整局；額外完整 forks/replays 仍須計入，不得看結果後加樣本。coverage 與其他 prerequisite gates 應在候選預註冊中固定。

1. 單元／native 檢查：動作與選擇索引、升級語意、成本、遺物、unknown fallback、RNG 無副作用、trace invariance、完整 legality/provenance、池排除與互不重疊。
2. Train：證明改到原本欲測的決策、有配對 afterstate 和完整 continuation；保留改善、退步與無效案例。
3. Probe：依固定 coverage、安全与 nonnegative-net gate 決定能否進新 Dev30。
4. Dev30：完整30對、正 net、所有 hard guards及適用 training-only retention 過關才進確認。依 Issue 的 H19+ 修訂，Dev p照報，但p<0.05不是cutoff；floor/HP下降是完整次要診斷，不能自行增設 veto。net≤0不進確認。
5. 確認：同一 frozen Candidate，兩組互不重疊的100配對，各組正 net、合計至少+10/200；完整報 exact one-sided sign-test，依 trial k 使用 `alpha_k=0.05/2^k`，不重置k、不看p後擴樣本。[Issue #19 現行契約](https://github.com/ericpeng0604-coder/sts-public-ci/issues/19)
6. 真實 STS1：確認通過後另做完整 mapping smoke、3-pair pilot、10-pair acceptance，以及可核對的 Ironclad A0 Act3 Victory。模擬器或 GitHub Actions成功不是實際遊戲验收。[live-game amendment](https://github.com/ericpeng0604-coder/sts-public-ci/issues/19)

每階段報 W/Games、C-only、P-only、net、discordant pairs、p、effective changes、Boss conversion、到Boss HP、完整 terminal和illegal/crash/timeout/communication。Boss conversion與HP只解釋機制；完整勝場仍是主要目標。
若進Boss的人群不同，conditional conversion也不能單獨拿來宣稱候選較強。

## 6. 工程範圍、工具與交付界線

本次唯一交付是此研究文件；不改 policy、PPO、G7、STS2，也不分配種子、重跑舊實驗或接付費模型。
未來若實作，限定 Issue #19 的 `scripts/sts1/**`、`src/roguelike_ai/sts1_phase3/**`、focused STS1 tests及 `evidence/sts1/g7-improvement/**`；範圍外入口要先按 Issue 規則登記。
優先使用 deterministic local tools；若日後引入 LLM，僅傳 compact DecisionContext、完整 canonical legal actions與最多一個相關 diff。完整 raw state、Episode、history保留在證據儲存，不當普通模型上下文。Provider usage本次 **NOT_MEASURED**。

研究方法：agent-reach 的 Exa search、GitHub CLI／API路由；四個專案固定commit及核心原碼交叉核對；另有背景研究代理獨立閱讀及更正後審核PASS。
沒有以搜尋摘要當已證實勝率，也沒有把作者自報成績當本地重現。
候選 `valiant-wjl/sts-rl-agent` 雖有搜尋索引描述，GitHub API/Raw在查閱時404，未納入已核對方案。
背景讀取 `ctx_fetch_and_index` 因缺 `turndown-plugin-gfm.cjs.js` 失敗，改用gh／Exa和context-mode sandbox處理。`agent-reach check-update` 因 executable不在PATH未完成，更新狀態 **NOT_VERIFIED**；不安裝或更新工具來中斷研究。

文件驗證：範圍、相對引用、固定版本與資料界線、112整局預算、git whitespace及原基底無改動核對。文件變更不重跑137項runtime測試，歷史通過不當作新候選驗證。
新策略選擇／實作、readiness API可行性、配對提升、正式確認、實際遊戲驗收皆 **NOT_RUN／NOT_VERIFIED**。

下一份實驗預註冊所需條件：新的Train證據、可核對Boss與升級／選牌入口、單一可解釋決策改動、完整法律動作／資源驗證、固定預算與互斥種子池。
