# CI/CD 流水线设置指南

## 已完成的工作

1. **创建了新分支**: `feature/ci-workflow`
2. **建立了GitHub Actions工作流**: `.github/workflows/ci.yml`
3. **添加了代码质量工具**:
   - flake8: 代码风格检查
   - black: 代码自动格式化
   - isort: 导入语句排序
   - pre-commit: 提交前钩子

## 工作流包含三个阶段

### 1. Test 测试阶段
- 支持Python 3.10和3.11版本
- 安装项目依赖
- 运行单元测试并生成覆盖率报告
- 代码风格检查和格式化验证

### 2. Build 构建阶段
- 构建Python包
- 上传构建产物作为Artifact

### 3. Deploy 部署阶段
- 仅在main分支触发
- 自动部署到PyPI (需要配置PyPI_TOKEN)

## 下一步需要手动配置的事项

### 1. 配置PyPI令牌 (可选)
如果需要自动部署到PyPI:
- 在GitHub仓库的Settings → Secrets and variables → Actions中添加`PYPI_TOKEN`
- 获取PyPI令牌: https://pypi.org/manage/account/token/

### 2. 安装pre-commit钩子 (推荐)
```bash
pip install pre-commit
pre-commit install
```

### 3. 推送分支到远程仓库
```bash
git push origin feature/ci-workflow
```

### 4. 创建Pull Request
在GitHub上创建从`feature/ci-workflow`到`main`的Pull Request，CI会自动运行

## 自定义工作流

可以根据项目需求修改`.github/workflows/ci.yml`文件:
- 添加更多Python版本测试
- 增加代码静态分析
- 部署到其他平台
- 添加Docker镜像构建
- 集成SonarQube代码质量分析

## 常用命令

```bash
# 运行测试
poetry run pytest

# 检查代码风格
poetry run flake8 src/ tests/

# 自动格式化代码
poetry run black src/ tests/
poetry run isort src/ tests/

# 运行pre-commit钩子
pre-commit run --all-files
```