// Homepage only (loaded from overrides/home.html): reveal-on-scroll and the install command's copy button.
(function () {
  "use strict";

  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var page = document.querySelector(".ds-page");
  if (!page) return;

  var revealEls = Array.prototype.slice.call(page.querySelectorAll("[data-reveal]"));
  if (!reduceMotion && "IntersectionObserver" in window) {
    page.classList.add("ds-js-reveal");
    // Plays each time an element scrolls in from below. Elements above the viewport stay shown, so scrolling up
    // never animates; an element resets only once it is fully off the bottom of the screen.
    var revealObserver = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting && entry.intersectionRatio < 0.1) return;
          var shown = entry.isIntersecting || entry.boundingClientRect.top < 0;
          entry.target.classList.toggle("is-visible", shown);
          // A jump (End key, restored scroll) skips elements without a callback; anything above a shown one is passed.
          if (shown) {
            for (var i = 0; revealEls[i] !== entry.target; i++) revealEls[i].classList.add("is-visible");
          }
        });
      },
      { threshold: [0, 0.1] }
    );
    revealEls.forEach(function (el) { revealObserver.observe(el); });
  }

  // The copy button ships hidden, so it only appears where it can work.
  var install = page.querySelector(".ds-install-command");
  if (install && navigator.clipboard && window.isSecureContext) {
    var copyBtn = install.querySelector(".ds-install-copy");
    var command = install.querySelector(".ds-install-text");
    var status = install.querySelector(".ds-install-status");
    var resetTimer;
    copyBtn.hidden = false;
    copyBtn.addEventListener("click", function () {
      clearTimeout(resetTimer);
      navigator.clipboard.writeText(command.textContent.trim()).then(
        function () {
          copyBtn.classList.add("copied");
          status.textContent = "Copied to clipboard";
          resetTimer = setTimeout(function () {
            copyBtn.classList.remove("copied");
            status.textContent = "";
          }, 2000);
        },
        function () {
          // Clipboard permission denied: select the command so it can be copied by hand.
          window.getSelection().selectAllChildren(command);
          status.textContent = "Couldn't copy automatically. The command is selected, copy it with your keyboard.";
        }
      );
    });
  }
})();
