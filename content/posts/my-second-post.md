+++
date = '2025-01-11T00:09:43+08:00'
<!-- draft = true -->
title = 'My Second Post'
+++
## Introduction

This is **bold** text, and this is *emphasized* text.

Visit the [Hugo](https://gohugo.io) website!

## 网站架构

主流静态网站框架有Hugo、Hexo、以及Github的亲儿子jekyll，最终我选择了Hugo框架，因为Hugo官网中的主题看起来更漂亮一点。

在挑选主题的时候，我在[LoveIt](https://themes.gohugo.io/themes/loveit/)和[PaperMod](https://themes.gohugo.io/themes/hugo-papermod/)之间犹豫了很久，甚至把两个主题都尝试搭了一遍，最终选择了更为简洁大方的PaperMod。

最终敲定的网站架构使用Github pages作为存储服务，并提供Web访问，Hugo作为静态博客框架，PaperMod作为Hugo的主题，并搭配Github Actions进行自动编译与发布。

发布一篇文章的流程是这样的：

1. 本地使用Markdown撰写一篇文章
2. 通过git同步至github仓库
3. Github Actions自动编译成静态站点并部署至Github Pages

## 配置Hugo

### 安装Git

Windows只需安装[Git for windows](https://git-scm.com/download/win)即可。

其它系统可以自行参考Git [官方文档](https://git-scm.com/book/zh/v2/%E8%B5%B7%E6%AD%A5-%E5%AE%89%E8%A3%85-Git) 安装。
### 安装Hugo