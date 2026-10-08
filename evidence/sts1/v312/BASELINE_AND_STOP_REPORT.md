# STS1 v3.12 loss-cause audit: baseline and stop decision

## 結論

**HOLD_PARENT（G7）；沒有可聲稱的勝率提升。** 已核對的 v3.11 Dev30 為 G7 與 Candidate 各 7/30 勝，無 discordant pairs，net 0、單尾 exact sign-test `p=1.0`。資料只支持一個待查的 Event 行為線索，無法把它連到 Dev30 的敗局；因此沒有足夠證據選定 v3.12 策略。階段 C、全新 Probe10、全新 Dev30 均 `NOT_RUN`，沒有新增遊戲局數。

## 範圍與版本

- Repo：`ericpeng0604-coder/sts-public-ci`；scope Issue [#19](https://github.com/ericpeng0604-coder/sts-public-ci/issues/19)。
- 本地工作 branch：`codex/sts1-g7-loss-cause-v312-20261007`；起始完整 SHA `c54bf019e44395a1e57d1a531bf382a1cc06ead8`。該 SHA 是 PR #18 的遠端 head；它不是已整合進 `main` 的證據。
- PR #18 仍是 Draft，base 是 snapshot branch `gpt/sts1-v311-audit-base-7921`（SHA prefix `7921a11a`）。它相對 `main` 的 1,077 commits ancestry 尚未整合；本報告不把 snapshot PR 當作 main 整合。
- Simulator pin：`7476a81954020087da31d41d16fddf475746ec2d`；G7 parent hash：`8313c99d9b0ab0c0d206fdd2f744fed11f4104d440dcb465cf7c78a517f9ccd0`；Candidate model hash：`f843ac19bb89f4e8e2703297cd728b3a97048801c343ddea656b17db85c43b7b`；Candidate adapter sidecar hash：`8e66a92c6462dd8365776d818278f8bf2e0e55ca9038a70cea0966d2d7431733`。

## 有界 evidence manifest

| Artifact | GitHub run | 大小 | SHA-256 |
|---|---:|---:|---|
| v3.11 初次 audit artifact | [37639357507](https://github.com/ericpeng0604-coder/sts-public-ci/actions/runs/37639357507) | 11,744,103 B | `2ecc2f4ae137c2d8e8c66c005c59c6b236410a25288bba593deded12fe53108f` |
| v3.11 recovery / Dev30 | [37641029190](https://github.com/ericpeng0604-coder/sts-public-ci/actions/runs/37641029190) | 11,757,847 B | `ef9a67f5f751c407abeea01fe9de1a384cd6ac091658f0d3af7fd14e3feeaeff` |
| v3.5 screen seeds | [37393165126](https://github.com/ericpeng0604-coder/sts-public-ci/actions/runs/37393165126) | 34,963 B | `098e469c27335baf9dc5e1b569fdeab023cd6c064a45490c781d017d1a77ff2f` |

三份下載檔 SHA-256 均與 GitHub 顯示的 artifact digest 一致。初次 audit run 因 `$GITHUB_OUTPUT` 格式錯誤而失敗；recovery run 成功並重用既有 Candidate，沒有重訓。初次與 recovery artifacts 內 Candidate model、sidecar 與 seed ledger 的 SHA-256 完全相同。

本次只讀取 `train300.txt`、`priority80.txt`、v36/v37/v38 的既有 Dev/Probe manifests、v3.11 ledger、候選摘要與 bounded decision records。**沒有讀取、解壓或執行 Gate/Fresh seed files；沒有啟動正式 Gate/Fresh 評估。**

## 階段 A：Dev30 paired baseline

以 recovery artifact 的 30 rows 和 `candidate/dev30-seeds.txt` 重算：

| 核對項目 | 結果 |
|---|---|
| Parent / Candidate | `7 / 30`、`7 / 30` |
| Candidate-only / Parent-only / net | `0 / 0 / 0` |
| 單尾 exact sign-test | `p=1.0`；0 discordant pairs 的既定值 |
| Pair 完整性 | 30/30；同一個 30-seed manifest，兩臂各自 simulator seed 相同 |
| Budget / simulator | 兩臂皆 MCTS-2000；同 pinned simulator |
| Safety / terminal | 兩臂 illegal、crash、timeout 均為 0；30 組皆為 complete run |

`pass=true` 表示 run 完整、安全且符合其檢查條件，不代表 Candidate 勝率較高或兩策略等效。唯一終局樓層差異在 seed `509222730`：Candidate 從 floor 41 到 floor 50，仍然敗局；final HP 兩臂都是 0，另多 26 game steps、80 MCTS actions、23 ArmG actions。其餘 29 組終局樓層相同；30 組 final HP 全相同。這是輔助結果，不是勝局，也不能單憑它推斷造成延後死亡的決策。

## 階段 B：activation 與可歸因敗因

### Candidate 行為

- Probe10 有 728 個非戰鬥決策，其中 477 個多選；獨立以 `parent + residual`（只在 gate allowed 時加一次）重算，與記錄的 parent top1、candidate top1 和 `top1_changed` **477/477 一致**。
- 28/477 次 gate allowed，實際 top1 只改變 1 次：Event、seed `645151561`、floor 8。Parent 選項與候選分數分別為 index 1、index 0；parent gap `0.041933`、相對 residual delta `0.050323`、改選後只領先 `0.008390`。
- 該 Probe10 的 Parent 與 Candidate 各 2/10 勝、0/0 discordant、net 0、`p=1.0`；10 局完整且安全。
- Teacher audit 的 62 個 target decisions / 56 seeds 全部 gate allowed；記錄有 15 次 top1 changes，其中 14 次改到 Teacher target、1 次改到另一選項。按 kind 的改選數為 Event 3/13、Shop 5/9、Card 3/15、Rest 1/3、Map 3/22。
- 以各自 alternative 的 `parent_gap(a)=parent(parent_top1)-parent(a)` 與 `relative_delta(a)=residual(a)-residual(parent_top1)` 重算，62 個 target 與 156 個 other-alternative pairs 均為 0 formula mismatch。Cluster summary 是 4 個跨 seed 重複 clusters、8 筆重複例（Event 6、Rest 2）；不能把同 seed 重複當作跨 seed 證據。

這些數字顯示訓練 audit 的 Teacher 改選多於遊戲中 Candidate 實際改選；Probe 的唯一改選也沒有帶來勝局差異。它們沒有證明 Event、Shop 或其他類別是 G7 的主要敗因。

Decision records 保存選項分數與 index；Teacher audit 只保存 legal choice count 和 descriptor SHA-256，沒有 canonical legal-action 描述。因此能列出唯一改選的 kind/seed/floor/index，不能可靠說出被選中的具體事件選項。該選項語義同樣是 `NOT_VERIFIED`。

### Dev30 敗局分類邊界

Dev30 summary 可核實 23 場敗局的 terminal floor，但沒有逐步 action trace、遭遇敵人、進戰 HP、牌組/遺物/藥水或路線決策。Parent 敗局 floor 分布為：`11:1, 16:2, 23:1, 24:1, 25:1, 33:9, 39:1, 41:2, 45:1, 47:1, 50:3`。Candidate 勝敗與 Parent 相同。

因此「某樓層死亡」只能作為線索，不能證明是戰鬥選擇、路線風險、牌組、休息/升級或資源使用造成。**G7 的主要可修正敗因、敵人與致敗決策均為 `NOT_VERIFIED`。** 現有 artifact 沒有足夠行動序列可支持兩到三個有實際敗局證據的 Stage C 假設。

## Seed ledger 與停止條件

依 recovery workflow 的切分邏輯及可讀的非正式 manifests 重算：

1. `train300` 300 個唯一 seeds；`priority80` 是其中 80 個；其餘 220 個。
2. v36 Dev30、v37 Dev50、v38 Probe30 共 110 個，互不重疊且在非-priority train set 內；workflow 因此得到 110 個 `remaining`。ledger 的 `remaining_untouched_count=110` 是 v39/v310/v3.11 seed allocation **之前**的數量，不是 v3.11 完成後剩餘數。
3. `remaining` 先配置 v39 Dev30、v310 Dev30，再配置 v3.11 Probe10 與 Dev30。v3.11 的 40 個 Probe/Dev seeds 彼此不重疊；之前取消且沒有執行遊戲的 30-seed reservation 不算已使用。
4. 以上已知 evaluation allocations 之後，在已核對的 train300 非-priority pool 中只剩 **10 個**未使用 IDs。這批屬於 train-pool，可作診斷/訓練用途，不能替代 40 個全新的獨立 Probe10 + Dev30 seeds。priority80 是既有 train pool，不能拿來冒充新 dev；Gate/Fresh 又明確禁止使用。

故目前可證的 seed pool 不足以完成 Stage E 的新 Probe10 + 不重疊 Dev30；也沒有完整敗局 trace 可支撐一個單一 Stage C intervention。依任務的 fail-closed 停止條件：

- Stage C train-pool games / Candidate training：`NOT_RUN`。
- 全新 Probe10 / Dev30：`NOT_RUN`；不重用 v3.11 observed seeds、不挪用 Gate/Fresh。
- Gate30/Gate50/Fresh100/Fresh500：`NOT_RUN`。
- 新增 game episodes：`0 / 140`。
- Champion：保留 G7；沒有程式、policy、retention 或安全閾值變更。

## 驗證、範圍與後續

- 本次為 artifact-only 診斷與本報告；未改動程式。Targeted `pytest` 未能啟動：系統 PATH 的 `python` 不可用，workspace bundled Python 沒有 `pytest`；該 Python 也沒有 `torch`，所以 Torch 專項測試亦為 `NOT_RUN`。未安裝套件，也未把既有 handoff 所述的測試結果冒充為本次測試。
- 程式 scope 未變；本次唯一預期新增檔案是此 Issue #19 允許的 `evidence/sts1/v312/BASELINE_AND_STOP_REPORT.md`。
- 下一步開始前需要可核對的、仍未使用且非 Gate/Fresh 的 Dev seed manifest；如沒有此 pool，依目前任務約束維持 HOLD_PARENT。要定位敗因則需要包含敗局決策序列及 encounter/deck/resource state 的診斷 artifacts；不得用 terminal floor 推造原因。
- 未宣稱 real-game、正式 Gate/Fresh、勝率提升或 main 整合證據。
