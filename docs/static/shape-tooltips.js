document.addEventListener("DOMContentLoaded", () => {
  // Nameless Returns use ZWSP-only labels so Sphinx can pair :returns:/:rtype:;
  // strip those labels so the HTML reads like "(Type) – description".
  document.querySelectorAll(".field-list li > p > strong").forEach((el) => {
    if (!/^\u200b+$/.test(el.textContent)) return;
    const next = el.nextSibling;
    if (
      next &&
      next.nodeType === Node.TEXT_NODE &&
      next.textContent.startsWith(" (")
    ) {
      next.textContent = next.textContent.slice(1);
    }
    el.remove();
  });

  tippy("abbr[title]", {
    content(el) {
      const text = el.getAttribute("title");
      el.removeAttribute("title");
      return text;
    },
    delay: [30, 0],
    animation: "shift-away-subtle",
    theme: "shape",
  });
});
