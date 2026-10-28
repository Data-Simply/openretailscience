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

document$.subscribe(() => {
  if (document.querySelector(".arithmatex") === null) return
  const script = document.createElement("script")
  script.src = "https://unpkg.com/mathjax@3.2.2/es5/tex-mml-chtml.js"
  document.head.appendChild(script)
})
