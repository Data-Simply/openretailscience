window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// MathJax costs seconds of main-thread time on mobile and only a few pages have math, so load it on those alone.
document$.subscribe(() => {
  if (document.querySelector(".arithmatex") === null) return
  const script = document.createElement("script")
  script.src = "https://unpkg.com/mathjax@3.2.2/es5/tex-mml-chtml.js"
  document.head.appendChild(script)
})
