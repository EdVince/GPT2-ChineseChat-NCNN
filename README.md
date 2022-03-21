# GPT2-ChineseChat-NCNN
**为什么要做**：最近在看去年2021年的tesla ai day，看下来获益良多，就是里面很多都需要爆炸的算力才能玩，但是我发现无论是tesla还是在场的开发者，貌似都对GPT这个模型很感兴趣，我就想着来玩一玩。

**做什么模型**：GPT有很多优秀的模型，选模型就让我头疼了好久，大家对于部署来说是更倾向于用[minGPT](https://github.com/karpathy/minGPT)这个模型的，但我没卡训一个中文的模型出来。所以只好选现成的中文模型，考虑到对话形式较好展示就选了[GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)这个项目了。(PS:经过测试这个模型对话质量并不高，只能图一乐，主要还是展示把GPT放到ncnn的工作)

**工作目标**：使用[ncnn](https://github.com/Tencent/ncnn)部署[GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)这个基于GPT2的中文对话模型，目标是给出x86和android端的demo

**PS**：最近各种工作都比较忙，而且工作主要也不用ncnn了。而且我要毕业了，忙着做毕设，所以就只有周末有时间做一下。考虑到GPT这种语言模型，与CV还是很不一样的，还没动手我就知道这东西不好做，但总要人来吃这个螃蟹的(不知廉耻的想要个star而已)


### 工作内容
- [ ] pytorch模型梳理与导出
- [ ] x86 demo
- [ ] android demo

### 参考
1. [ncnn](https://github.com/Tencent/ncnn)
2. [GPT2-chitchat](https://github.com/yangjianxin1/GPT2-chitchat)