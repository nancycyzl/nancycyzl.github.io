---
layout:       post
title:        "Publish blogs with Github pages"
author:       "Nancycy"
header-style: text
catalog:      true
mathjax:      true
tags:
    - 
---

Use template from https://huangxuan.me/

## Steps for setup

1. download all file from template
2. create a github repository with name [username].github.io
3. delete unnecessary files and modify `_config.yml` and other settings
4. open https://nancycyzl.github.io/ in web browser to check

## Steps to publish a post

1. complete a note in Obsidian (as local save)
2. copy the note to `_\post` folder and rename as `yyyy-mm-dd-tile.md`
3. put related images to `img` folder
4. in the file, add the yaml setting
5. in the file, check formulars: insert lines before and after `$$` symbols
6. in the file, modify the image embedding method as html code
7. push the changes to github
8. wait a moment and check on web browser

## Things to note

### Filename requirement

Change the post file name in `yyyy-mm-dd-[title].md` format.

### Yaml setting

Need include yaml setting at the front of each post.
```
---  
layout:       post  
title:        "NSGA-II"  
author:       "Nancycy"  
header-style: text  
catalog:      true  
mathjax:      true  
tags:  
    - optimization  
---
```

### Latex and Mathjax

The template does not support latex, so it uses Mathjax to enable formulars. Need to modify `_inludes/mathjax_support.html` file.

Original code is:
```json
tex2jax: {  
  inlineMath: [ ['$','$'] ],  
  displayMath: [ ['$$','$$'] ],  
  processEscapes: true,  
}
```
Change to
```json
tex2jax: {  
  inlineMath: [ ['$','$'] , ["\\(","\\)"]],  
  displayMath: [ ['$$','$$'] , ["\\[","\\]"]],  
  processEscapes: true,  
}
```

> [!NOTE]
> For the display mode (formular block), there must be empty lines before and after the formular.

### Image handling

The template uses `img` folder to hold all images. So when publish a post, copy corresponding images into the `img` folder.

Also, do not use Obsidian code as `![[file]]`, use html code as 
```html
<img src="/img/post2024/NSGAII_procedure.png" alt="image" width="704">
```

> [!NOTE]
> If there are any spaces in the path, use `%20` to replace each space.


