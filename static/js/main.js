document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initPresets();
  initTagToggles();
  initTechFilter();
  initBookmarks();
});

function initTheme() {
  const toggleBtn = document.getElementById("themeToggle");
  const storedTheme = localStorage.getItem("projectforge_theme");
  const systemPrefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;

  const currentTheme = storedTheme || (systemPrefersDark ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", currentTheme);
  updateThemeIcon(currentTheme);

  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const activeTheme = document.documentElement.getAttribute("data-theme");
      const newTheme = activeTheme === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", newTheme);
      localStorage.setItem("projectforge_theme", newTheme);
      updateThemeIcon(newTheme);
    });
  }
}

function updateThemeIcon(theme) {
  const iconSpan = document.getElementById("themeIcon");
  const textSpan = document.getElementById("themeText");
  if (iconSpan) {
    iconSpan.textContent = theme === "dark" ? "☀️" : "🌙";
  }
  if (textSpan) {
    textSpan.textContent = theme === "dark" ? "Light" : "Dark";
  }
}

function initPresets() {
  const presetChips = document.querySelectorAll(".preset-chip");
  const form = document.getElementById("recommendationForm");
  if (!form) return;

  presetChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const bg = chip.dataset.background;
      const domains = JSON.parse(chip.dataset.domains || "[]");
      const tech = JSON.parse(chip.dataset.tech || "[]");
      const diff = chip.dataset.difficulty;
      const scope = chip.dataset.scope;
      const interests = chip.dataset.interests;

      const bgSelect = document.getElementById("academicBackground");
      if (bgSelect && bg) bgSelect.value = bg;

      const diffSelect = document.getElementById("difficultySelect");
      if (diffSelect && diff) diffSelect.value = diff;

      const scopeSelect = document.getElementById("scopeSelect");
      if (scopeSelect && scope) scopeSelect.value = scope;

      const interestsInput = document.getElementById("interestsInput");
      if (interestsInput && interests) interestsInput.value = interests;

      const domainCheckboxes = document.querySelectorAll('input[name="domains"]');
      domainCheckboxes.forEach(cb => {
        cb.checked = domains.includes(cb.value);
        updateTagLabelState(cb);
      });

      const techCheckboxes = document.querySelectorAll('input[name="tech_stack"]');
      techCheckboxes.forEach(cb => {
        cb.checked = tech.includes(cb.value);
        updateTagLabelState(cb);
      });

      form.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
}

function initTagToggles() {
  const tagLabels = document.querySelectorAll(".tag-checkbox-label");
  tagLabels.forEach(label => {
    const cb = label.querySelector('input[type="checkbox"]');
    if (!cb) return;

    label.addEventListener("click", () => {
      setTimeout(() => updateTagLabelState(cb), 10);
    });
  });
}

function updateTagLabelState(checkbox) {
  const label = checkbox.closest(".tag-checkbox-label");
  if (!label) return;
  if (checkbox.checked) {
    label.classList.add("selected");
  } else {
    label.classList.remove("selected");
  }
}

function initTechFilter() {
  const filterInput = document.getElementById("techFilterInput");
  if (!filterInput) return;

  filterInput.addEventListener("input", (e) => {
    const query = e.target.value.toLowerCase().trim();
    const techItems = document.querySelectorAll(".tech-item");
    const categoryGroups = document.querySelectorAll(".tech-category-group");

    techItems.forEach(item => {
      const name = item.dataset.techName || "";
      if (!query || name.includes(query)) {
        item.style.display = "inline-flex";
      } else {
        item.style.display = "none";
      }
    });

    categoryGroups.forEach(group => {
      const visibleItems = group.querySelectorAll('.tech-item[style*="display: inline-flex"]');
      const allItems = group.querySelectorAll(".tech-item");
      if (query && visibleItems.length === 0) {
        group.style.display = "none";
      } else {
        group.style.display = "block";
      }
    });
  });
}

function initBookmarks() {
  const bookmarkBtns = document.querySelectorAll(".bookmark-btn");
  const savedIds = getSavedBookmarks();

  bookmarkBtns.forEach(btn => {
    const id = btn.dataset.projectId;
    if (savedIds.includes(id)) {
      btn.classList.add("bookmarked");
      btn.innerHTML = "★ Saved";
    }

    btn.addEventListener("click", (e) => {
      e.preventDefault();
      toggleBookmark(id, btn);
    });
  });
}

function getSavedBookmarks() {
  try {
    return JSON.parse(localStorage.getItem("projectforge_bookmarks") || "[]");
  } catch (e) {
    return [];
  }
}

function toggleBookmark(id, btn) {
  let saved = getSavedBookmarks();
  if (saved.includes(id)) {
    saved = saved.filter(x => x !== id);
    btn.classList.remove("bookmarked");
    btn.innerHTML = "☆ Save";
  } else {
    saved.push(id);
    btn.classList.add("bookmarked");
    btn.innerHTML = "★ Saved";
  }
  localStorage.setItem("projectforge_bookmarks", JSON.stringify(saved));
}

function copyShareLink() {
  navigator.clipboard.writeText(window.location.href).then(() => {
    alert("Project link copied to clipboard!");
  }).catch(() => {
    prompt("Copy this URL:", window.location.href);
  });
}
