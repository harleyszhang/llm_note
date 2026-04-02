# 6-龙虾 OpenClaw 笔记整理

> 整理时间：2026-04-03
> 来源：小红书、知乎、GitHub、博客园、什么值得买等平台热门内容
> 筛选标准：内容质量、实用性、系统程度、社区热度

---

## 目录

1. [OpenClaw 是什么](#1-openclaw-是什么)
2. [核心架构与记忆系统](#2-核心架构与记忆系统)
3. [安装与部署](#3-安装与部署)
4. [模型配置与选型指南](#4-模型配置与选型指南)
5. [渠道接入（多平台）](#5-渠道接入多平台)
6. [Skills 技能系统](#6-skills-技能系统)
7. [安全防护](#7-安全防护)
8. [成本控制策略](#8-成本控制策略)
9. [实用技巧与最佳实践](#9-实用技巧与最佳实践)
10. [学习资源汇总](#10-学习资源汇总)
11. [参考来源](#11-参考来源)

---

## 1. OpenClaw 是什么

OpenClaw（龙虾）是一个开源、自托管的 AI Agent 网关，你可以把它部署在自己的电脑或服务器上，让它成为你 24 小时在线的个人助理。

### 与 ChatGPT 的本质区别

| 维度 | ChatGPT | OpenClaw |
|------|---------|----------|
| 角色定位 | 顾问（你问它答） | 员工（主动执行任务） |
| 数据归属 | 存储在 OpenAI 服务器 | 完全在你自己的设备 |
| 平台接入 | 仅 ChatGPT 界面 | 20+ 消息平台 |
| 持久记忆 | 有限的会话记忆 | 四层记忆系统，长期学习 |
| 可扩展性 | 有限 | ClawHub 13,000+ Skills |
| 成本结构 | 订阅制 | 软件免费，自付 API 费用 |

### 核心能力

- **生活自动化**：接管邮件、日历、消息管理；浏览网页、填写表单、数据抽取
- **多平台接入**：支持 WhatsApp、Telegram、Slack、Discord、飞书、钉钉、QQ、微信等 20+ 平台
- **电脑操作**：安装软件、开发程序、监控任务、执行 Shell 命令
- **智能记忆**：四层记忆系统，越用越懂你

### 项目基本信息

| 属性 | 详情 |
|------|------|
| 开源协议 | MIT（完全免费） |
| GitHub Stars | 60,000+（2026年3月） |
| 贡献者 | 1,075+ |
| 推荐版本 | v2026.3.8 |
| 官网 | [openclaw.ai](https://openclaw.ai) |
| GitHub | [github.com/openclaw/openclaw](https://github.com/openclaw/openclaw) |

---

## 2. 核心架构与记忆系统

### 2.1 Gateway（网关）

Gateway 是 OpenClaw 的核心枢纽，默认监听 `ws://127.0.0.1:18789`，负责：

- 统一接收来自所有消息平台的消息
- 将消息路由给 AI 模型进行处理
- 将 AI 的响应发回对应平台
- 管理 Skills 的加载与执行

### 2.2 四层记忆系统（核心特性）

| 层级 | 文件 | 说明 | 持久性 |
|------|------|------|--------|
| 第一层 | SOUL.md | Agent 的核心人格与行为准则 | 永久 |
| 第二层 | TOOLS.md | 可用工具列表 | 自动维护 |
| 第三层 | USER.md | 用户偏好与习惯 | 长期 |
| 第四层 | Session Memory | 当前会话上下文 | 临时 |

> **关键理解**：SOUL.md 是 Agent 的「灵魂文件」，决定了 AI 的行为风格。四层记忆系统是 OpenClaw 和所有聊天机器人最本质的区别——ChatGPT、Claude 都是"踹一脚它动一下"，而 OpenClaw 有自主记忆和持续学习能力。

### 2.3 配置文件

- `openclaw.yaml`：渠道配置文件，定义各消息平台的连接参数
- `~/.openclaw/openclaw.json`：核心配置文件，定义模型提供商、Agent 行为等

---

## 3. 安装与部署

### 3.1 环境要求

| 依赖 | 最低版本 |
|------|---------|
| Node.js | >= 22 |
| npm | 最新版 |
| 操作系统 | macOS / Linux / Windows |
| 内存 | 建议 2GB 以上 |

### 3.2 标准安装（三步完成）

```bash
# 第一步：全局安装
npm install -g openclaw@latest

# 第二步：引导式初始化
openclaw onboard --install-daemon

# 第三步：诊断检查
openclaw doctor
```

> ⚠️ **安全警告**：务必使用官方包名 `openclaw`，不要安装任何带前缀的变体。2026年3月曾出现恶意 npm 包伪装成官方安装器。

### 3.3 非研发人员一键安装

Mac/Linux：
```bash
curl -fsSL https://clawd.org.cn/install.sh | bash -s -- --registry https://registry.npmmirror.com
```

Windows：
```powershell
iwr -useb https://clawd.org.cn/install.ps1 -OutFile install.ps1; ./install.ps1 -Registry https://registry.npmmirror.com
```

### 3.4 部署方案对比

| 方案 | 适合人群 | 优点 | 缺点 |
|------|---------|------|------|
| 本地 npm | 开发者 | 免费灵活、调试方便 | 需保持电脑开机 |
| Docker | 有 Docker 经验 | 环境隔离、可移植 | 需 Docker 知识 |
| 云服务器（推荐） | 大多数用户 | 24/7 在线、成本低 | 需配置服务器 |
| 一键平台 | 非技术用户 | 最简部署 | 灵活度低 |

### 3.5 云服务器方案推荐

| 方案 | 月费 | 说明 |
|------|------|------|
| 阿里云轻量服务器 | ¥6-9/月 | 新用户优惠，最受欢迎 |
| 腾讯云 Lighthouse | ¥8-12/月 | 社区支持好 |
| Fly.io | 免费起步 | 适合轻度使用 |

---

## 4. 模型配置与选型指南

### 4.1 国际模型

| 模型 | 输入价格/1M | 输出价格/1M | 上下文 | 定位 |
|------|------------|------------|--------|------|
| Claude Opus 4.6 | $5.00 | $25.00 | 200K | 最强推理 |
| Claude Sonnet 4.6 | $3.00 | $15.00 | 200K | 主力模型（推荐） |
| Claude Haiku 4.5 | $1.00 | $5.00 | 200K | 轻量高速 |
| GPT-5.4 | $2.50 | $15.00 | 272K | OpenAI 旗舰 |
| Gemini 3 Flash | $0.50 | $3.00 | 免费 | 适合心跳/Cron |

### 4.2 国产模型

| 模型 | 输入价格/1M | 输出价格/1M | 定位 |
|------|------------|------------|------|
| DeepSeek-V3.2.2 | $0.14 | $0.28 | 性价比之王 |
| GLM-5 | $0.80 | $2.56 | 国产最强代码 |
| GLM-4.7-Flash | 免费 | 免费 | 轻量免费 |
| Qwen 3.5 Coder | $0.22 | $1.00 | 代码专用 |

### 4.3 国产模型购买链接

| 厂商 | 首月价格 | 购买链接 |
|------|---------|---------|
| 智谱 GLM | ¥49 | [bigmodel.cn](https://www.bigmodel.cn/glm-coding) |
| Kimi | ¥49 | [kimi.com/code](https://www.kimi.com/code) |
| 阿里通义 | ¥7.9（首月） | [aliyun.com](https://www.aliyun.com/benefit/ai/aistar) |
| 字节豆包 | ¥9.9 | [volcengine.com](https://www.volcengine.com/activity/codingplan) |

### 4.4 关键配置概念

- **内置 Provider**：Anthropic、OpenAI、Google、智谱等无需额外配置
- **自定义 Provider**：DeepSeek、豆包等需手动添加
- **Fallback 机制**：主模型不可用时自动切换备选（省钱核心策略）
- **`models.mode: "merge"`**：保留内置 Provider 同时叠加自定义配置（务必设置）

---

## 5. 渠道接入（多平台）

### 5.1 推荐接入顺序

| 梯队 | 平台 | 耗时 | 推荐理由 |
|------|------|------|---------|
| 第一梯队 | Telegram、QQ、微信 | 5-10 分钟 | 不需公网 IP，扫码即用 |
| 第二梯队 | Discord、飞书 | 15-20 分钟 | 文档齐全，内置支持 |
| 第三梯队 | WhatsApp、Slack、钉钉 | 25-40 分钟 | 社区方案成熟 |
| 第四梯队 | iMessage | 需额外条件 | 需特殊硬件 |

### 5.2 Telegram 接入（推荐入门）

```bash
# 1. 在 Telegram 搜索 @BotFather，发送 /newbot
# 2. 获取 Bot Token
# 3. 写入 openclaw.yaml：
```

```yaml
channels:
  telegram:
    enabled: true
    botToken: "YOUR_BOT_TOKEN"
    dmPolicy: pairing
```

### 5.3 国内平台

- **QQ**：腾讯官方支持，扫码 1 分钟绑定
- **微信**：微信 → 设置 → 插件 → 「微信 ClawBot」，扫码连接
- **飞书**：OpenClaw 2026.2 起原生内置支持
- **钉钉**：社区插件，Stream 模式免公网
- **企业微信**：Agent 模式 / Bot 模式两种接入

> **核心建议**：大部分渠道使用 long-polling 或 Stream 模式，bot 主动连接服务器，不需要公网 IP。只有 webhook 回调场景才需要暴露公网。

---

## 6. Skills 技能系统

### 6.1 三层优先级

| 优先级 | 位置 | 说明 |
|--------|------|------|
| 最高 | `<workspace>/skills/` | 项目级，仅当前工作区 |
| 中 | `~/.openclaw/skills/` | 用户级，全局生效 |
| 最低 | bundled skills | 内置 55 个 Skills |

### 6.2 必装 Top 10 Skills

| 排名 | Skill | 用途 |
|------|-------|------|
| 1 | Gmail / Google | 邮件收发、日历管理 |
| 2 | Agent Browser | 浏览器自动化 |
| 3 | Summarize | 视频/网页/邮件自动摘要 |
| 4 | GitHub | 仓库管理、PR 审查 |
| 5 | Claude Code | MCP 桥接 Claude Code 能力 |
| 6 | Web Search | 联网搜索实时信息 |
| 7 | File Manager | 本地文件操作 |
| 8 | Calendar | 日程管理 |
| 9 | Translator | 多语言翻译 |
| 10 | Image Gen | AI 图片生成 |

### 6.3 Skill 安全必装

- **skill-vetter**：扫描安装的 Skill 是否安全
- **self-improvement**：让 AI 自我改进和学习
- **vector-memory**：向量记忆搜索，提高记忆准确性

### 6.4 ClawHub 技能市场

| 指标 | 数据 |
|------|------|
| 总注册技能 | 13,729 |
| 精选技能 | 5,494 |
| 被标记为恶意 | 800+（约 20%） |

> ⚠️ **严重警告**：ClawHub 质量问题非常严重。安装第三方 Skill 前务必查看源码，推荐参考 [awesome-openclaw-skills](https://github.com/VoltAgent/awesome-openclaw-skills) 精选列表。

### 6.5 自建 Skill 最小结构

```
my-skill/
├── SKILL.md          # 必须，Skill 核心定义
├── scripts/          # 可选，辅助脚本
└── templates/        # 可选，模板文件
```

---

## 7. 安全防护

### 7.1 ClawHavoc 供应链攻击事件（2026年1-2月）

| 指标 | 数据 |
|------|------|
| 恶意 Skills | 341 个初步确认，800+ 后续发现 |
| 受影响设备 | 135,000+ |
| 攻击手法 | 植入信息窃取木马，篡改 SOUL.md「洗脑」Agent |

### 7.2 安全红线

- ❌ 拒绝任何要求"下载 zip 文件""执行 shell 脚本""输入密码"的 Skill
- ✅ 安装前用 skill-vetter 扫描
- ✅ 定期检查 SOUL.md 内容
- ✅ 配置配对码（dmPolicy: pairing）
- ✅ 多人环境中使用 VM 或专用服务器

### 7.3 常用安全命令

```bash
openclaw doctor          # 全面诊断检查
openclaw backup create   # 创建完整备份
openclaw backup verify   # 验证备份完整性
```

---

## 8. 成本控制策略

### 8.1 核心省钱技巧

1. **Fallback 机制**：主模型挂了自动切备选
2. **Prompt Caching**：降低重复上下文成本达 90%
3. **Batch API**：Claude Batch API 享 50% 折扣
4. **分层用模型**：日常用 Sonnet/DeepSeek，复杂任务才用 Opus
5. **心跳用免费模型**：Gemini Flash 免费额度适合心跳和定时任务
6. **国产模型优先**：DeepSeek 输入 $0.14/M，极致低价

### 8.2 版本更新渠道

| 渠道 | 命令 | 适合人群 |
|------|------|---------|
| stable | `openclaw update --channel stable` | 大多数用户 |
| beta | `openclaw update --channel beta` | 尝鲜用户 |
| dev | `openclaw update --channel dev` | 开发者 |

---

## 9. 实用技巧与最佳实践

### 9.1 新手快速上手路线

1. 安装 OpenClaw → 配置国产模型 API（推荐 DeepSeek 或阿里首月 ¥7.9）
2. 接入 Telegram 或 QQ（5分钟）
3. 安装 3-5 个必装 Skill（browser、web-search、file-manager）
4. 日常对话中训练 AI 了解你的习惯
5. 逐步扩展更多 Skill 和平台

### 9.2 记忆训练技巧

- OpenClaw 越用越懂你，关键是多在日常对话中让它了解你的偏好
- 定期查看和编辑 SOUL.md、USER.md，强化 AI 对你的理解
- 使用 vector-memory Skill 解决复杂上下文记忆不准确的问题

### 9.3 节约 Token 的方法

- 不要一次性安装太多 Skills（增加 system prompt 长度）
- 心跳和定时任务用免费模型（Gemini Flash、GLM-4.7-Flash）
- 设置合理的 session pruning 策略
- 使用 Fallback 链避免昂贵的重试

### 9.4 远程访问方案

```bash
# Tailscale（官方推荐）
tailscale serve --bg https+insecure://127.0.0.1:18789    # 仅内网
tailscale funnel --bg https+insecure://127.0.0.1:18789    # 公网

# SSH 端口转发
ssh -L 18789:127.0.0.1:18789 user@your-server
```

---

## 10. 学习资源汇总

### 官方资源

| 资源 | 链接 |
|------|------|
| 官方文档 | [docs.openclaw.ai](https://docs.openclaw.ai) |
| GitHub 仓库 | [github.com/openclaw/openclaw](https://github.com/openclaw/openclaw) |
| ClawHub 技能市场 | [clawhub.com](https://clawhub.com) |
| Discord 社区 | [discord.gg/clawd](https://discord.gg/clawd) |

### 中文社区与教程

| 资源 | 链接 | 说明 |
|------|------|------|
| OpenClaw 中文社区 | [clawd.org.cn](https://clawd.org.cn) | 中文社区门户 |
| 学习资源聚合 | [openclaw101.dev/zh](https://openclaw101.dev/zh) | 聚合教程 |
| 中文教程合集 | [awesome.tryopenclaw.asia](https://awesome.tryopenclaw.asia/) | 从新手到中级 |
| awesome-openclaw-skills | [GitHub](https://github.com/VoltAgent/awesome-openclaw-skills) | 5,000+ 精选 Skills |
| awesome-openclaw-usecases | [GitHub](https://github.com/hesamsheikh/awesome-openclaw-usecases) | 使用案例集 |

### 国产模型部署文档

| 厂商 | 部署文档 |
|------|---------|
| 智谱 | [docs.bigmodel.cn](https://docs.bigmodel.cn/cn/coding-plan/tool/openclaw) |
| Kimi | [platform.moonshot.ai](https://platform.moonshot.ai/docs/guide/use-kimi-in-openclaw) |
| 阿里 | [bailian.console.aliyun.com](https://bailian.console.aliyun.com) |
| 字节 | [volcengine.com](https://www.volcengine.com/docs/82379/2183190) |

---

## 11. 参考来源

本笔记整理自以下 5 篇热门优质内容：

1. **知乎 - 保姆级教程：无成本零门槛安装配置 OpenClaw 龙虾 AI 全能助手**
   - 来源：[知乎专栏](https://zhuanlan.zhihu.com/p/2012626753718342006)
   - 亮点：零基础友好、云服务器获取指南、国内网络适配

2. **知乎 - OpenClaw（小龙虾）使用技巧全攻略：从配置到记忆训练、安全防护**
   - 来源：[知乎专栏](https://zhuanlan.zhihu.com/p/2018032078357315812)
   - 亮点：记忆训练方法、安全防护、Token 节约技巧、高级策略

3. **LightNote - OpenClaw 完全教程：从入门到精通**
   - 来源：[blog.lightnote.com.cn](https://blog.lightnote.com.cn/openclaw-complete-tutorial-from-beginner-to-expert)
   - 亮点：最全面系统、架构解析透彻、多平台接入详解、模型价格全景

4. **博客园 - OpenClaw 保姆级教程**
   - 来源：[cnblogs.com](https://www.cnblogs.com/zh94/p/19670571)
   - 亮点：国产模型购买链接汇总、一键安装脚本、必装 Skill 推荐、社区资源聚合

5. **GitHub - OpenClaw 官方仓库**
   - 来源：[github.com/openclaw/openclaw](https://github.com/openclaw/openclaw)
   - 亮点：权威一手信息、架构图、安全机制、完整功能列表

---

> 💡 **一句话总结**：OpenClaw 不只是一个聊天机器人，而是一个运行在你自己设备上的、有记忆、能执行任务、接入 20+ 平台的个人 AI 助理。零成本安装，按需选择模型，从 Telegram 5分钟接入开始，逐步解锁它的全部潜力。
