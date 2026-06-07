# 职业转型 + 搬迁研究：Microsoft L64 → model-facing LLM 工程师

> **这是什么**：一份决策级研究报告，针对在职 Microsoft L64（China STCA）资深云工程师，转型 model-facing LLM 工作（fine-tuning / distillation / agent / evals）+ 跨区搬迁。整合了四轮多 agent 研究（Search MCP fan-out → 3 票对抗式核查 → 综合）。
>
> **方法**：4 个 workflow，累计 ~196 个 agent、~13.7M token、~1,575 次工具调用。报告一（三地市场+三原型，173 claim/杀 1）、报告二（MS 内部 transfer + L-1，88 claim/杀 2）、报告三（北欧/爱尔兰/澳新 model-facing + 签证，153 claim/杀 6）、报告四（MS L64 各地 comp，杀 3）。被杀/存疑 claim 在 §8 单列，正文不当事实引用。
>
> **语言例外**：本仓库 CLAUDE.md 规定 in-repo notes 用英文。此文件是个人职业研究（非 curriculum/code），且为忠实保留薪资/签证数字避免再翻译引入错误，正文保留中文，术语/公司名/签证项目/货币/level 用英文。
>
> **生成/更新**：2026-06-06
>
> **当事人画像（已确认事实）**：
> - NTU 校友 · 在职 Microsoft **L64（Senior SDE 顶格）** · China **STCA 6 年** · 10+ 年总经验
> - 工作内容：面向客户的 microservices on **Azure Kubernetes (AKS)**，**非 Azure 产品线**（跨 org）；是 AKS 的*使用者*非*构建者*
> - 深 systems/perf 背景（.NET allocator/GC、NativeMemory、NUMA、async）· 无 ML PhD · 在 GX10 上自学 fine-tuning
> - **目标工作**：model-facing（和 LLM 本身打交道：fine-tuning / distillation / agent / evals，不挑），**明确不要 GPU 调度 / inference-serving 性能 / cluster plumbing**
> - **搬迁偏好顺序**：1) 北欧（Oslo/Copenhagen）2) Ireland（Dublin）3) Singapore/NZ/Australia。**美国最低优先级。**
> - **路径**：Microsoft 内转优先 + 外部本地雇佣备选
> - **收入底线**：现 ~1,000,000 RMB/yr 税前 ≈ **€128K**；要 +30-50% gross（≈ €167-192K），或 net/PPP 跑通的前提下接受更低名义值

---

## 0. 一句话结论

**最诚实的结论：你的搬迁偏好顺序（北欧 > 爱尔兰 > 澳新/坡）和薪酬顺序、岗位可得性顺序几乎完全倒挂。** 以 Microsoft L64 内转身份，**没有任何一个你偏好的目的地能在 gross 口径干净越过 +30% floor（€167K）**。

按你的两个硬约束（model-facing + 偏好北欧）联合筛选，落在三个现实选项：

1. **Copenhagen（北欧里唯一站得住的）** —— 满足你北欧偏好 + 是**非美唯一有实锤 LLM fine-tuning/RLHF 在招岗**的 Microsoft site。代价：L64 gross €153-164K（+20-28%），但丹麦 55.9% 顶税把 net 压到 ~€85-90K，相对你当前 net **基本打平**。是"用平薪换北欧生活 + 转 model 岗"，不是涨薪。**Forskerordning 35% 平税**是唯一能改写净值的杠杆。
2. **Dublin（偏好与收入最平衡的折中）** —— Microsoft EMEA 重镇（6000+ 人），有 Copilot/Azure AI Foundry 的 model-facing 岗（偏应用）。L64 gross €160-163K（+25-27%，差一口气到 +30%），但 41.5% 税 → net ~€93-95K。签证最顺（ICT→CSEP→Stamp 4）。
3. **Singapore（net 最优，但偏好排最后 + 岗位存疑）** —— gross 名义不达标，但 24% 封顶税让 **net ~€128-131K 全场最优**，几乎正好对齐你当前到手。NTU 学位让 EP/COMPASS 很顺。短板：L64 数据薄（派生估计）、model 岗多为 Azure AI 方向存疑。

**划掉**：Oslo（只有 backend 岗 + €133K 几乎打平）、Stockholm（无产品工程 DC，内转路径不存在）、Sydney（gross 不达标 + model 岗存疑）、Auckland（无 site，数据极弱）。

**一个可能比薪资更早收敛选择的约束**：Microsoft 真正的核心模型岗（pre/post-training、RLHF、RL systems）几乎**全在美国**。非美 site 里只有 **Copenhagen 有 model-facing 实锤**，Dublin 偏应用。如果"做真正的模型工作"权重高于"非美地点"，这个约束比薪酬更先决定你的选择——值得你单独排一次序。

---

## 1. 收入底线现实核对（最关键，先讲）

基线：现 ~1,000,000 RMB/yr 税前 ≈ **€128K**。底线 +30% = **€167K**，+50% = **€192K**（gross 口径）。

> **口径说明**：薪资保留源币种，EUR 折算用 2026-06 即期 **0.86 EUR/USD**（报告三误用过时的 1.09，US 等值偏高 6-8%，已在下表修正）。net 为单身、无养老金抵扣、各地 2026 税率粗算。"vs €128K"按 **gross** 比（你的 floor 是 gross 定义的）。

### 主表 — Microsoft L64 by location（已用 L64 数据 + 0.86 汇率修正）

| City | MS 工程/AI site？ | model-facing 岗 | L64 gross (源币 + ≈EUR) | 实际税 | est. net/yr | vs €128K gross | 过 +30-50%？ |
|---|---|---|---|---|---|---|---|
| **Copenhagen** | ✅ Dev Center 800+ 人 | ✅ **实锤 LLM/RLHF/Applied Sci** | $178K 中位 / $191K(L64样本) ≈ **€153-164K** | ~55.9% 顶档 | ~€85-90K | **+20-28%** | ⚠️ 仅上沿勉强 |
| **Dublin** | ✅ 6000+ 人，EMEA 重镇 | ✅ Copilot/AI Foundry（偏应用） | **€160-163K** TC（直接EUR计） | ~41.5% | ~€93-95K | **+25-27%** | ❌ 差一口气 |
| **Singapore** | ✅ APAC 工程 site | ⚠️ MSRA(research门槛)/Azure AI | SGD 210-240K ≈ **€141-161K**（派生估计） | ~16-18% | **~€128-131K** | +10-26% | ❌ gross不够 **net最优** |
| **Oslo** | ⚠️ Dev Center 390 人 | ⚠️ M365 backend（非微调） | NOK 1.57M ≈ **€133K** 中位 | ~47.4% | ~€80-88K | **+4%（打平）** | ❌ 名义打平/小减 |
| **Sydney** | ⚠️ 偏销售/CE | ⚠️ 存疑（偏 Azure AI） | A$227K ≈ **€140K** | ~33.7% | ~€93K | +9% | ❌ |
| **Auckland** | ❌ 未列 MS locations | ❌ 存疑 | NZ$297K ≈ €152K（数据极弱） | ~34.7% | ~€98K | +17% | ❌ 数据不可外推 |
| **Stockholm** | ❌ **无产品工程 DC** | ❌ 路径不存在 | 无 MS 数据（市场 senior ~€68K） | ~52% | — | 路径不存在 | ❌ |
| *Redmond（US锚）* | ✅ 总部，核心模型岗在此 | ✅ 全部 | $270K TC ≈ €232K | ~32-35%，WA无州税 | ~€159K | +54% | ✅ 但你已 deprioritize |
| *SF Bay（US锚）* | ✅ | ✅ | $266K ≈ €229K | +CA ~10% | ~€151K | +79% | ✅ 但州税侵蚀 |

**这一节的硬话**：gross 口径下你的 +30% floor 几乎全军覆没（含 Dublin 差一口气）；**net 口径下只有 Singapore 实际达标**（net ≈ 当前 net 甚至略高），Dublin/Copenhagen 接近打平。**去北欧是生活方式决策，不是收入决策。**

### Forskerordning（丹麦研究者税制）—— Copenhagen 的胜负手

合格者前 7 年享 27% + 8% AM-bidrag = **35% 平税**（2026-01 起最低月薪门槛降到 DKK 65,400，你必然达标）。普通税率下 DKK 1.1M gross → net ~€85K；**Forskerordning 35% 下 → net ~€96K**。这是 Copenhagen 财务上变得可接受的唯一场景。资格条件（过去 10 年非丹麦税务居民）须向税务顾问确认。

---

## 2. model-facing 岗位现实（你在拿 hireability 换"做想做的事"）

前情：前序报告结论 ML-systems/training-infra 对你迁移度 **HIGH**；applied/model-facing **MEDIUM 且 entry 更饱和**；research（PhD-gated）**closed**。你选 model-facing，这一份尊重选择并量化代价。

### 你主动放弃了最强差异化卖点

- **ML-infra 因"生产约束下的分布式系统调试"长期供给不足**——正是你 systems 背景的天然优势区。
- **applied fine-tuning 因 Python/PyTorch 易自学，申请池大、更饱和。**
- 所以选 model-facing = 放弃你的护城河。**这是真实成本。** 缓解：别往 entry 挤（entry 仅占 foundation-model 岗 ~3%、每岗上千申请者），用你的资历直接打 **mid/senior applied**。

### 四类 model-facing 岗对你的可达性排序

1. **AI/LLM Agent Engineer — 最可达**。本质 product eng + orchestration（tool-calling、RAG、MCP/A2A、生产可靠性），**不需要 ML training**，直接吃你 AKS/分布式/可靠性背景。
2. **Evals / model-behavior — 可达**。需统计严谨 + systems thinking，无 PhD 要求，多从 MLOps/backend 转入。
3. **Applied fine-tuning（product-company 级）— 需拿得出手的作品，但不要 PhD**。Anthropic Applied AI Finetuning Engineer 要 3+ 年训练/微调 + 2+ 年 Solutions，不列 PhD；Microsoft Copilot Tuning Applied Scientist：Bachelor's+4y / Master's+3y / **PhD+1y（非必需）**。
4. **Distillation / frontier-lab research engineer — 最不可达**（PhD-gated：Scale AI LLM Eval RS 要 NeurIPS/ICML 发表；Airbnb Principal LLM Fine-tuning 要 PhD）。对你关闭。

### 三个缓解杠杆

1. **用内转绕过冷申请墙**（内转成功率 70-80% vs 外部冷投 20-30%；Anthropic senior 冷投 2026 回应率 <1%）。Microsoft 内部确有 PhD 非必需的 model-facing org：**Copilot Tuning**、**Applied AI Engineer II**（agentic/RAG/evals，Bachelor's+2y，**JD 明说 cloud/AKS 背景直接相关**）、**MAI**（含 SFT/RL）。
2. **建 fine-tuning portfolio 作为"等价经验"**（几乎所有 product JD 把 PhD 写成"or equivalent experience"；Anthropic 明示作品可替代文凭）。诚实时间线：在职 part-time 15-20h/周，**12-18 个月**才到 mid-level 真正有竞争力（不是 bootcamp 宣传的 3-6 个月）。
3. **把 systems 背景当桥不要丢**：你的 GX10 自学若产出 LoRA/QLoRA + FSDP/DeepSpeed 实测作品，**同时补 model-facing 入场券和保留 infra 差异化**——最高 ROI 方向，也最贴合你"flexible across fine-tuning/distillation/agent/evals"的自陈。

---

## 3. 按偏好排序的地区分析

### 3.1 Nordics（第一偏好，岗位/薪酬这组最弱）

**Copenhagen（北欧唯一站得住）**
- **site & 内转 & model 岗**：MS Dev Center Copenhagen（Lyngby，800+ 人）有**实锤的 LLM fine-tuning / RLHF / agentic 在招岗**（[LinkedIn JD](https://www.linkedin.com/jobs/view/4341015688/)）——这是非美 site 里**唯一**与你"转 model-facing"直接对得上的实锤。
- **签证（中国籍）**：EU ICT Directive 丹麦 opt-out，走 **Pay Limit Scheme**（2026 门槛 DKK 552,000，轻松达标）或 **Fast-Track**；**MLE 在 2026 Positive List 上**（无薪资下限工签）。
- **pay**：L64 gross €153-164K（+20-28%），但 55.9% 顶税 → net ~€85-90K（**Forskerordning 35% → ~€96K**）。相对当前 net 打平到小正。
- **结论**：北欧唯一同时有真实 DC + model 岗 + 像样薪酬带的点。"为生活方式付费"，不涨薪。

**Oslo（划掉）**
- MS Dev Center Norway（390 人）真实但是 **M365 backend**，非微调/蒸馏。L64 中位 €133K（+4%，几乎打平，个别样本低于现状）。$6.2B Norway 投资是数据中心算力，不是工程 headcount。
- **签证修正**（报告三证据曾错）：Norway **有** ICT/assignment 子路线，但对永居**更不利**（6 年上限、不计永居、境内不可转）。若去，要让 Microsoft **本地雇佣**（标准技术工签）而非 assignment。
- **结论**：内转只能拿 backend，pay 几乎打平。不建议。

**Stockholm（划掉）**：Kista 的 591 人实体是销售/咨询（SNI 62201），**无产品工程 DC**，内部产品岗调动路径不存在。SEK 33.7B 投资是数据中心。

### 3.2 Ireland / Dublin（第二偏好，最平衡折中）

- **site & 内转 & model 岗**：6000+ 人 EMEA 重镇。2024-11 IDA 官宣 **550 个研发岗**（含 Applied Scientist / AI SWE，明确做 LLM fine-tuning/agents/evals，横跨 M365 Copilot / Azure AI Foundry）；2025-12 仍在招明写 LLM fine-tuning 的 AI SWE II 和 Agent Cloud 的 Senior AI SWE。**是你内转就能落到 model-facing req 的目的地（偏应用）。**
- **签证（中国籍，这组最顺）**：ICT Employment Permit（门槛 €49,523，海外满 6 个月，你已满足）。**两个 ICT 坑**：不累积永居、配偶不自动获工作权。**解法**：境内转 **CSEP**（software/AI 在 Critical Skills 清单，门槛 €40,904 你远超，无 LMT，4-6 周）→ 2 年转 **Stamp 4**（永居等价，配偶得 Stamp 1G 工作权）→ 5 年入籍。
- **pay**：L64 gross €160-163K（+25-27%，差一口气到 +30%），41.5% 税 → net ~€93-95K，叠加 Dublin 1-bed €2,000-2,400/月，购买力提升温和。**SARP** 可对 €100K 以上 30% 免税最长 5 年，值得查。
- **注意**：2025 下半年 Dublin 裁了 ~250 人，site 收缩期，内转可能拿"新 hire 指引带"而非 refresher 包。
- **外部 fallback**：OpenAI Dublin（FDE $220-280K）、Google Cloud Dublin（GenAI FDE）。

### 3.3 Singapore（第三梯队，但 net/税务其实很强）

- **site & 内转 & model 岗**：APAC HQ，商业实体内转能进但**非 model-facing**；真正 model-facing 在 **MSRA Singapore**（在招 Foundation Model/LLM Researcher，**但 research track 需研究履历，对你（无 PhD/发表）门槛高**）。
- **签证（中国籍，NTU 给结构性优势）**：EP + COMPASS。**NTU QS #12 → C2 拿满 20 分**；ML 在 Shortage Occupation List。senior ML 薪资不到 SGD 22,500/月豁免线须过 COMPASS，但 NTU + 竞争力薪资 + 非中资公司中国籍占比通常可过。
- **pay**：L64 gross 派生估计 €141-161K（gross 不达标），但 24% 封顶税 → **net ~€128-131K 全场最优**，几乎正好对齐当前 net。房租最高（1-bed SGD 3,000-4,500/月）。
- **外部雇主强**：NVIDIA、Google DeepMind、OpenAI、Apple、ByteDance 均在 SG 有 ML 岗。

### 3.4 Australia / New Zealand（第三梯队尾部）

- **Sydney**：site 偏销售/CE，SWE headcount 少；L64 gross A$227K ≈ €140K（+9%，不达 floor）；net ~€93K 不算差但 gross 不够；model 岗存疑（偏 Azure AI）。**签证**：subclass 482（2025-12 更名 Skills in Demand），Specialist Skills stream 门槛 A$141,210，有 482→186 永居路径。
- **New Zealand**：**Microsoft 无实质工程 site（内转无门）**；senior MLE 仅 NZD 138-160K（≈€75-87K，**明确降薪**）；AEWV 2025-03 起取消中位工资下限。**只能外部本地雇佣，且这组最大降薪。**

---

## 4. 内转 vs 外部，分地区

内转只在「Microsoft 有相关 site **且** 有 model-facing req」时成立：

| 地区 | MS 有相关 site？ | 内转拿到 model-facing？ | 真实路径 |
|---|---|---|---|
| **Copenhagen** | ✅ Dev Center | **✅ 实锤 LLM/RLHF 岗** | **内转可行（北欧首选）** |
| **Dublin** | ✅ 6000+ | **✅（偏应用）** | **内转可行（最平衡）**；OpenAI/Google 作 fallback |
| Singapore | ✅ APAC + MSRA | ⚠️ 仅 MSRA，research 门槛挡你 | 商业实体内转=非 model-facing；model-facing 走外部或冲 MSRA |
| Oslo | ✅ 但 backend | ❌ 基本否 | 内转只能拿 backend |
| Sydney | ✅ 但偏销售 | ⚠️ 不明显 | 外部本地雇佣为主 |
| Stockholm | ❌ 无 DC | — | 路径不存在 |
| New Zealand | ❌ | — | 只能外部本地雇佣 |

**核心**：你"内转 + model-facing"两愿望能同时满足的地方，实质只有 **Copenhagen（实锤）和 Dublin（偏应用）**。其余要 model-facing 就得走外部本地雇佣，承担更低成功率。

---

## 5. 修正后的行动路径

### 阶段 0（现在 → 6 个月）：建 portfolio + 占住内转资格

- **Portfolio（最高 ROI）**：用 GX10 产出 2-3 个 documented 作品，覆盖你 flexible 的范围且复用 systems 优势：
  1. **QLoRA/LoRA 微调**（NF4 + bf16），带内存预算推导 + 显存实测（`torch.cuda.memory._record_memory_history`）+ 前后 evals。← curriculum **Track A1/A6/A7**
  2. **FSDP/DeepSpeed（ZeRO-1/2/3 + offload）** 训练跑通，写清 GX10 统一内存下 offload 带宽（273 GB/s 共享）权衡。**同时是 model-facing 入场券 + infra 差异化展示。**
  3. **agent + evals** 小项目（tool-calling/RAG + 可复现 eval harness），对口 Microsoft Applied AI Engineer JD。← curriculum **Track A9/A10**
- 公开化：GitHub + 1-2 篇技术博客（Anthropic 明示可替代文凭的"equivalent experience"）。
- **内转资格**：ICT 要求的"海外 6 个月在职"你 6 年早满足。现在开始内部接触 **Copenhagen Dev Center** 和 **Dublin Copilot/Azure AI Foundry / Applied AI Engineer / Copilot Tuning** 的 hiring manager（暖引荐 >> 冷申请）。

### 阶段 0 的三个事实核实

- [ ] 当前 Rewards 评分稳在 **100+**、不在 PIP（否则 2025-04 起禁止内部转岗）。
- [ ] 当前岗位在岗满 12-18 个月（进扩张 AI org 可能例外）。
- [x] ~~海外 1 年（ICT/L-1 闸门）~~ —— **STCA 6 年早满足。**

### 阶段 1（6-12 个月）：主攻 Copenhagen 或 Dublin 内转

- **若 model-facing 实质 > 地点**：主攻 **Copenhagen**（非美唯一 model 实锤）。走 Pay Limit Scheme / Positive List + **Forskerordning 35% 平税**（确认过去 10 年非丹麦税务居民）。睁眼接受平薪。
- **若偏好+收入平衡 > 纯 model**：主攻 **Dublin**。目标 req：AI SWE II / Senior AI SWE (Agent Cloud) / Copilot Tuning Applied Scientist。**谈判盯 level + RSU**：L64（€160-163K）才稳，避免 down-level 到 L62/L63 低带。走 ICT→CSEP→Stamp 4。
- **若只认 net**：**Singapore** 是强候选（net 全场最优，NTU 让 EP 顺），但接受 model 岗多为 Azure AI 方向、core model 岗要冲 MSRA。

### 阶段 2（12-18 个月）：到"mid-level 真正有竞争力"临界点

portfolio 起效，能打 mid/senior applied。届时内转若成 → 落地；若 req 一直没开 → 用 portfolio 打 Dublin 外部（OpenAI/Google）或 Singapore 外部（NVIDIA/Google/OpenAI）。

**一句话路径**：先在 GX10 上把 fine-tuning + FSDP/DeepSpeed + agent/evals 作品做出来并公开 →（model 实质优先）暖引荐进 Copenhagen Dev Center 的 LLM 岗，走 Pay Limit + Forskerordning；（偏好+收入平衡）进 Dublin model-facing req，走 ICT→CSEP→Stamp 4 → 睁眼接受北欧/爱尔兰都是名义平薪到小涨、为生活方式与"做想做的事"付费，而非涨薪。

---

## 6. 三个最大风险

1. **偏好与现实倒挂（最大）**：最想去的北欧岗位最少、薪酬最低。缓解=认清 Copenhagen 是北欧唯一可行点，Oslo/Stockholm 划掉；接受"为生活方式付费"的框架。
2. **model-facing = 放弃 infra 护城河**：你最强的差异化在 ML-infra，选 model-facing 是主动放弃。缓解=portfolio 同时展示 FSDP/DeepSpeed（保留 infra 信号）+ fine-tuning（model 入场券）；别往 entry 挤，打 mid/senior。
3. **核心模型岗几乎全在美国**：非美 site 多为应用/Azure AI 方向。缓解=认清 Copenhagen 是非美唯一 model 实锤；若"做真正模型工作"权重最高，可能要重新权衡是否接受美国（你当前排最低）。

---

## 7. Leveling 校正（重要，影响锚点）

确认：Microsoft IC 阶梯 SDE(L59-60) → SDE II(L61-62) → **Senior SDE(L63-64)** → Principal(L65-67) → Partner(L68-69) → Distinguished(L70+)。**L64 = Senior 顶格，Principal 从 L65 起**（2025-05 泄露指引逐字证实）。跨公司 **L64 ≈ Google L5 ≈ Meta E5 ≈ Amazon L6**（Senior 段）。

> 前序报告曾把"L64-65 Principal"写糊——已更正：你 L64 是 Senior 顶档、Principal 下面一级。各地"senior ML"市场价对的是你 L64，别锚到 staff/principal。已知向上偏移：外部 Meta E5/Google L5 转 MS 常落 L65 而非 L64。

---

## 8. 置信度与存疑项（不要过度信任）

**已被对抗性核查杀掉（killed，勿作事实引用）**：
- ❌ "Microsoft Oslo L63 中位 NOK 1,026,360 ≈ €89K"——1,026,360 是全 level 中位非 L63；真实 L63 Oslo ~NOK 1.19M ≈ €110K（且用了过时汇率）。
- ❌ "Norway 没有 ICT 路线"——推翻，Norway 有 assignment/ICT 子路线，对永居更不利（见 §3.1）。
- ❌ "Denmark senior SWE 中位 DKK 860K" / "DK ML/AI $88,993"——过时，真实 senior ~DKK 814K、ML/AI ~$109K。
- ❌ "L64 avg TC $339K（range $295-420K）"——虚高/混入 sign-on，用 levels.fyi 结构化中位 $270K。
- ❌ "Blind 称 Redmond L64 TC $350K"——$350K 实为发帖人自己（Google/Nutanix）薪酬，误读。
- ❌ 报告一："Google/Meta/Apple Senior ML $350-420K、RSU 40-60%"——实测 Meta E5 中位 $440-511K、Google L5 ~$401K，RSU ~38-49%。
- ❌ 报告二：MAIST HPC Engineer / LLM Inference 的部分 comp floor 数字（用官方 JD 为准）。

**低置信/存疑（hedge）**：
- ⚠️ **Singapore L64 是派生估计**（SGD 210-240K 由 L62 + 级差推算，非观测）。floor 判断据此精度有限。
- ⚠️ **Auckland**：未列 MS locations；L64 数据仅 2024 两个 14-15 年工龄样本，不可外推。
- ⚠️ **Stockholm**：无任何 MS L64 数据；"路径不存在"基于 site 结构证据。
- ⚠️ **Oslo**：levels.fyi 仅 2 个样本，中位 €133K 置信度低。
- ⚠️ **Denmark MS 全 level 中位样本极小、快照间漂移**——Denmark 薪资当方向性非精确。
- ⚠️ **非美 model-facing 岗**：除 Copenhagen 实锤外，Dublin/SG/Sydney/Oslo 多为存疑或偏 Azure AI 应用，非核心模型研发。
- ⚠️ **泄露薪酬带是"新 hire 指引"**，员工反映偏低；2025 加薪冻结 + 股票池缩减意味着内转更可能拿新 hire 带而非 refresher——取数偏保守是合理的。
- ⚠️ **SARP / Forskerordning 个人资格**（尤其"过去 10 年非该国税务居民"）须向税务顾问确认，是 Ireland/Denmark 净值关键变量但非自动适用。
- ⚠️ **内转 70-80% vs 冷投 20-30%、entry 占 3%、12-18 月时间线**——单一 LinkedIn 分析估计，方向可信、精确数字勿当定论。
- ⚠️ **汇率**：EUR 折算按 2026 年中 0.86 EUR/USD 近似；波动会改变"是否过线"的边际判断。RMB→EUR 用你 pin 的 €128K 基准。

**结构性可信（survived，可放心用）**：Microsoft leveling 阶梯（L64=Senior 顶格、Principal 从 L65）、levels.fyi 各地 L64/Senior SDE 中位、Dublin 550 岗官宣 + 在招 LLM JD、Copenhagen LLM/RLHF 在招 JD、各国签证官方门槛（IE ICT/CSEP、DK Pay Limit/Positive List、SG EP/COMPASS、AU 482、NO UDI）、Stockholm 无产品 DC、各地税率官方数。

---

## 9. Sources

**Leveling / 跨公司映射**
- https://www.resumeadapter.com/companies/microsoft/levels
- https://dataconomy.com/2025/07/30/leaked-microsoft-pay-docs-reveal-up-to-408k-salaries/
- https://www.businessinsider.com/microsoft-pay-guidelines-new-hire-offers-exception-2025-7
- https://www.teamblind.com/post/microsoft-equivalent-of-meta-e5-08fsdcxz

**US 锚点（Redmond / SF Bay）**
- https://www.levels.fyi/companies/microsoft/salaries/software-engineer/levels/64
- https://www.levels.fyi/companies/microsoft/salaries/software-engineer/levels/senior-sde/locations/greater-seattle-area
- https://www.levels.fyi/companies/microsoft/salaries/software-engineer/levels/senior-sde/locations/san-francisco-bay-area

**Ireland / Dublin**
- https://www.levels.fyi/companies/microsoft/salaries/software-engineer/levels/64/locations/ireland
- https://www.idaireland.com/latest-news/press-release/an-taoiseach,-simon-harris-welcomes-microsoft%E2%80%99s-decision-to-deliver-550-new-irish-based-engineering
- https://www.irishjobs.ie/job/ai-software-engineer/microsoft-job106098461
- https://www.ziprecruiter.ie/jobs/479397335-senior-ai-software-engineer-at-microsoft
- https://careers.microsoft.com/v2/global/en/locations/dublin.html
- https://businessplus.ie/jobs/microsoft-new-jobs/
- https://www.irishtimes.com/business/2025/11/28/microsoft-ireland-has-cut-250-jobs-since-last-summer/
- https://www.gov.ie/en/department-of-enterprise-tourism-and-employment/publications/intra-company-transfer-employment-permits/
- https://enterprise.gov.ie/en/what-we-do/workplace-and-skills/employment-permits/employment-permit-eligibility/highly-skilled-eligible-occupations-list/
- https://home-affairs.ec.europa.eu/policies/migration-and-asylum/eu-immigration-portal/intra-corporate-transferee-ict-ireland_en
- https://aftertax.ie/
- https://kpmg.com/ie/en/insights/tax/budget-2026/tables.html
- https://www.tax121.com/blog/ireland-income-tax-complete-guide-for-2025-2026
- https://getmyhomereport.ie/dublin-rental-prices
- https://openai.com/careers/forward-deployed-engineer-dublin/

**Nordics（Oslo / Copenhagen / Stockholm）**
- https://www.levels.fyi/companies/microsoft/salaries/software-engineer/levels/64/locations/norway
- https://www.levels.fyi/companies/microsoft/salaries/software-engineer/levels/senior-sde/locations/denmark
- https://www.microsoft.com/da-dk/development
- https://www.linkedin.com/jobs/view/4341015688/
- https://en.wikipedia.org/wiki/Microsoft_Development_Center_Norway
- https://nyidanmark.dk/uk-UA/Words-and-concepts/SIRI/Minimum-amounts/Pay-Limit-Scheme's-minimum-amount
- https://travunited.com/blog/updates-from-denmark-positive-lists-foreign-workers-path-will-be-easier-in-2026
- https://home-affairs.ec.europa.eu/policies/migration-and-asylum/eu-immigration-portal/intra-corporate-transferee-ict-denmark_en
- https://www.udi.no/en/important-messages/new-salary-levels-from-1-september-2025/
- https://www.udi.no/en/want-to-apply/work-immigration/skilled-workers/
- https://taxsummaries.pwc.com/denmark/individual/taxes-on-personal-income
- https://www.skatteetaten.no/en/Rates/Maximum-effective-marginal-tax-rates/
- https://www.hitta.se/verksamhet/fkpbdhnpq
- https://www.svt.se/nyheter/inrikes/microsoft-miljardsatsar-pa-ai-i-sverige

**Singapore / Australia / New Zealand**
- https://www.levels.fyi/companies/microsoft/salaries/software-engineer/locations/singapore
- https://www.levels.fyi/companies/microsoft/salaries/software-engineer/levels/62/locations/singapore
- https://www.iras.gov.sg/taxes/individual-income-tax/basics-of-individual-income-tax/tax-residency-and-tax-rates/individual-income-tax-rates
- https://www.microsoft.com/en-us/research/group/microsoft-research-asia-singapore/opportunities/
- https://www.mom.gov.sg/passes-and-permits/employment-pass/eligibility
- https://www.ntu.edu.sg/about-us/facts-figures/university-rankings
- https://www.levels.fyi/companies/microsoft/salaries/software-engineer/levels/senior-sde/locations/australia
- https://www.levels.fyi/companies/microsoft/salaries/software-engineer/levels/64/locations/new-zealand
- https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/temporary-skill-shortage-482/salary-requirements
- https://careers.microsoft.com/v2/global/en/locations.html

**model-facing 岗位现实 / 跨区 / 汇率**
- https://www.linkedin.com/pulse/breaking-foundation-model-engineering-unfiltered-2025-nishad-aliyar-qi94f
- https://www.linkedin.com/jobs/view/applied-ai-engineer-ii-at-microsoft-4416664585
- https://www.linkedin.com/jobs/view/senior-applied-scientists-and-principal-applied-scientists-multiple-positions-copilot-tuning-at-microsoft-4334254312
- https://builtin.com/job/applied-ai-finetuning-engineer/3279331
- https://microsoft.ai/careers/
- https://www.mtfxgroup.com/tools/historical-currency-exchange-rates/usd-to-eur-rate/

---

## 附录 A：美国路径（你已 deprioritize，保留备查）

你把美国排最低，故移至附录。但前两份研究的美国 + L-1 分析有保留价值，**若你日后因"核心模型岗几乎全在美国"而重新考虑美国**，要点如下：

- **L-1 绕开 H-1B 抽签**：你 STCA 6 年满足"境外 1 年"，内转走 L-1 无配额无抽签（外部应聘要进 H-1B ~35% 抽签）。你几乎一定是 **L-1B（specialized knowledge）**（senior IC），**L-1B 无 EB-1C 免 PERM 快车道**，中国籍 EB-2/EB-3 PERM 排期多年。Microsoft 适用 blanket L（I-129S）。先例：2024-05 微软给 700-800 名中国 AI/cloud 工程师 US/Ireland/AU/NZ relocation。
- **美国 model-facing 岗最全**：CoreAI、MAIST、Copilot Tuning、Applied AI Engineer 全在美国，PhD 非必需。
- **美国 L64 comp**：Redmond $270K TC ≈ €232K（WA 无州税，net ~€159K，+54%）；SF Bay $266K ≈ €229K（+CA 税）。**远超你 floor**——若你能接受美国，收入维度完胜北欧/爱尔兰。这是你偏好与收入倒挂的极端例证：你最不想去的地方钱最多、岗位最全。
