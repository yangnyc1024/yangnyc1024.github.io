---
layout: post
title: "Writing Setup for This Site"
description: "A short note about using Markdown posts with GitHub Pages."
tags:
  - GitHub Pages
  - Markdown
  - Writing
---
This site now supports Markdown-based writing through GitHub Pages and Jekyll.

To publish a new post, add a file under `_posts/` with a name like:

```text
2026-04-14-my-new-post.md
```

Start the file with front matter:

```yaml
---
layout: post
title: "My New Post"
tags:
  - LLMs
  - RAG
---
```

After that, you can write the rest of the article in normal Markdown.

This setup is a good fit for technical and research writing because it keeps the source lightweight, versioned, and easy to edit over time.
