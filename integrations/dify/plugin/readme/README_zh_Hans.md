## memmachine-plugin

**作者：** memverge
**版本：** 0.2.1
**类型：** tool

### 描述

## 面向 AI Agent 的通用记忆层

认识 MemMachine：一个为高级 AI Agent 打造的开源记忆层。它使 AI 应用能够从过往会话中学习、存储并召回数据与偏好，从而丰富后续交互。MemMachine 的记忆层可跨多个会话、多个 Agent 以及不同的大语言模型持续存在，逐步构建一个精细且持续演进的用户画像。它让 AI 聊天机器人转变为更个性化、具备上下文感知能力的 AI 助手，以更高的准确性与更强的深度来理解并响应。

## MemMachine 适合谁？

- 构建 AI Agent、智能助理或自治工作流的开发者。
- 研究 Agent 架构与认知模型的研究人员。

## 关键特性

- **多种记忆类型：** MemMachine 支持 Working（短期）、Persistent（长期）以及 Personalized（画像/档案）等记忆类型。
- **开发者友好的 API：** 提供 Python SDK、RESTful 与 MCP 接口与端点，让你可以轻松将 MemMachine 集成到你的 Agent 中。更多信息请参阅
  [API 参考指南](https://docs.memmachine.ai/api_reference)。

## 架构

1. 通过 API 层与 Agent 交互
	用户与 Agent 交互，Agent 通过 RESTful API、Python SDK 或 MCP Server 连接到 MemMachine 的记忆核心。
2. MemMachine 管理记忆
	MemMachine 处理交互，并将其存储为两类不同的记忆：用于对话上下文的 Episodic Memory（情景记忆）以及用于长期用户事实的 Profile Memory（画像记忆）。
3. 数据持久化到数据库
	记忆会持久化到数据库层：Episodic Memory 存储在图数据库中，Profile Memory 存储在 SQL 数据库中。

<div align="center">

![MemMachine Architecture](https://raw.githubusercontent.com/MemMachine/MemMachine/main/assets/img/MemMachine_Architecture.png)

</div>

## 用例与示例 Agent

MemMachine 的多样化记忆架构可应用于任何领域，将通用机器人升级为专业化、专家级的智能助理。我们不断增长的
[examples](../../../examples/README.md) 列表展示了记忆驱动的 Agent 的无限可能，它们可以集成到你自己的应用与解决方案中。

- **CRM Agent：** 你的 Agent 能召回客户的完整历史与当前商机阶段，主动帮助销售团队建立关系并更快促成成交。
- **Healthcare Navigator（医疗导航）：** 提供持续的患者支持；Agent 能记住病史并跟踪治疗进展，带来更顺畅的医疗体验。
- **Personal Finance Advisor（个人理财顾问）：** Agent 会记住用户的资产组合与风险偏好，基于其完整历史给出个性化的金融洞察。
- **Content Writer（内容写作）：** 构建一个能记住你的写作风格指南与术语的助理，确保所有文档始终保持一致性。

我们很期待看到你正在构建的项目。欢迎加入
[Discord 服务器](https://discord.gg/usydANvKqD)，并在 **showcase** 频道分享你的项目。

