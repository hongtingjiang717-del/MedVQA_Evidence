# 如何把本项目发到GitHub（AutoDL终端）

## 第1步：创建自己的GitHub仓库
登录GitHub，右上角`+` → `New repository`，名称建议 `MedVQA-Evidence`，首次建议选择 **Private**；不要勾选自动创建README、.gitignore等初始化选项，因为压缩包已包含这些文件。

## 第2步：在AutoDL中上传本压缩包
将压缩包上传到 `/root/autodl-tmp/`，在终端执行：

```bash
cd /root/autodl-tmp
unzip -o MedVQA-Evidence_GitHub_Ready.zip
cd MedVQA-Evidence_GitHub_Release
python day5_app.py --check
python -m unittest discover -s tests -v
```

若实际ZIP解压结构不同，执行`ls`确认`README.md`所在的目录再`cd`进去。

## 第3步：先核查提交文件

```bash
git init
git branch -M main
git add .
git status
```

确认没有原始医疗数据、大模型权重、`*.safetensors`、缓存目录、`.env`、`results/day4_validation.jsonl`。仅需保留`results/day4_validation_summary.json`这个聚合摘要。

## 第4步：提交及关联远程仓库

```bash
git config user.name "你的GitHub用户名"
git config user.email "你的GitHub提交邮箱"
git commit -m "Initial release: MedVQA-Evidence"
git remote add origin https://github.com/<你的用户名>/MedVQA-Evidence.git
git push -u origin main
```

将`<你的用户名>`替换成自己的真实GitHub用户名。远程登录时推荐GitHub CLI、凭据管理器或个人访问令牌（PAT），不要把令牌直接粘贴进命令或发到聊天记录。

**GitHub仓库可见性**：Private仓库只有你授权的人能看，面试展示可在审查后改成Public；公开前先确认影像素材及依赖的授权。

## 第5步：发布后的检查
打开GitHub仓库，检查README内两张截图正常显示、`results/day4_validation_summary.json`能点开、没有大模型权重和隐私文件。再将链接写到简历的项目名称旁。
