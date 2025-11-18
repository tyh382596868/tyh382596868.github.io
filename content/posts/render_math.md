+++
date = '2025-11-18T19:44:05+08:00'
draft = false
title = 'hugo Render math'
+++
```
config.yml 里启用了 Goldmark 的 renderer.unsafe 和 extensions.passthrough，允许 $$...$$/$...$/\[...\] 等分隔符在 Markdown 中原样输出，避免被转义
```

```yaml
markup:
  goldmark:
    renderer:
      unsafe: true
    extensions:
      passthrough:
        enable: true
        delimiters:
          block:
            - - "\\["
              - "\\]"
            - - "$$"
              - "$$"
          inline:
            - - "\\("
              - "\\)"
            - - "$"
              - "$"
```
新增 layouts/partials/math.html，通过 PaperMod 的 extend_head.html 自动注入 KaTeX CSS/JS，并在页面加载后调用 renderMathInElement 统一渲染数学公式，同时忽略 pre/code 等标签避免误渲染

layouts/partials/math.html
```html
{{- $katexVersion := "0.16.11" -}}
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@{{ $katexVersion }}/dist/katex.min.css" crossorigin="anonymous" />
<script defer src="https://cdn.jsdelivr.net/npm/katex@{{ $katexVersion }}/dist/katex.min.js" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@{{ $katexVersion }}/dist/contrib/auto-render.min.js" crossorigin="anonymous"></script>
<script>
  document.addEventListener("DOMContentLoaded", function () {
    if (typeof renderMathInElement !== "function") {
      return;
    }
    renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true },
      ],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"],
      throwOnError: false,
    });
  });
</script>

```

themes/PaperMod/layouts/partials/extend_head.html
```html
{{- /* Head custom content area start */ -}}
{{- /*     Insert any custom code (web-analytics, resources, etc.) - it will appear in the <head></head> section of every page. */ -}}
{{- /*     Can be overwritten by partial with the same name in the global layouts. */ -}}
{{- /* Head custom content area end */ -}}
{{ if or .Params.math .Site.Params.math }}
{{ partial "math.html" . }}
{{ end }}
```