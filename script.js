const themeToggle = document.getElementById("theme-toggle");
const body = document.body;

const savedTheme = localStorage.getItem("theme");

if (savedTheme === "light") {
  body.classList.add("light-mode");
  themeToggle.textContent = "☀️";
} else {
  themeToggle.textContent = "🌙";
}

themeToggle.addEventListener("click", () => {
  body.classList.toggle("light-mode");

  if (body.classList.contains("light-mode")) {
    localStorage.setItem("theme", "light");
    themeToggle.textContent = "☀️";
  } else {
    localStorage.setItem("theme", "dark");
    themeToggle.textContent = "🌙";
  }
});

const tocContainer = document.getElementById("post-toc");
const articleEl = document.getElementById("post-article");

if (tocContainer && articleEl) {
  const headings = Array.from(articleEl.querySelectorAll("h2")).filter((h) => h.id);

  if (headings.length > 0) {
    const title = document.createElement("p");
    title.className = "post-toc-title";
    title.textContent = "On this page";
    tocContainer.appendChild(title);

    const list = document.createElement("ul");
    const links = [];

    headings.forEach((heading) => {
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = "#" + heading.id;
      link.textContent = heading.textContent;
      item.appendChild(link);
      list.appendChild(item);
      links.push({ id: heading.id, link });
    });

    tocContainer.appendChild(list);

    if ("IntersectionObserver" in window) {
      const setActive = (id) => {
        links.forEach(({ id: linkId, link }) => {
          link.classList.toggle("active", linkId === id);
        });
      };

      const observer = new IntersectionObserver(
        (entries) => {
          const visible = entries.filter((entry) => entry.isIntersecting);
          if (visible.length > 0) {
            setActive(visible[0].target.id);
          }
        },
        { rootMargin: "-100px 0px -70% 0px" }
      );

      headings.forEach((heading) => observer.observe(heading));
    }
  }
}