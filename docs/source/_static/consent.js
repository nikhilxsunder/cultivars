/* Storage preferences for the cultivars docs.
 *
 * Builds a floating button and a right-hand drawer, persists the visitor's
 * choice in localStorage, and gates Google Analytics through Consent Mode.
 * pydata-sphinx-theme already loads gtag.js with every consent signal set to
 * "denied" (its `analytics` option does this); nothing here loads a script.
 * The only side effects are the two gtag('consent', 'update', ...) calls and,
 * on revocation, best-effort deletion of the _ga cookies.
 *
 * Runs with `defer`, so the DOM is parsed when it executes. No dependencies.
 */
(function () {
  "use strict";

  var STORAGE_KEY = "cultivars.consent";
  var VERSION = 1;
  var DEFAULTS = { analytics: true };

  var CATEGORIES = [
    {
      id: "essential",
      title: "Essential",
      locked: true,
      description:
        "Required for the site to work: the light/dark mode you pick and " +
        "this consent record. Nothing here identifies you or leaves your " +
        "browser.",
      disclosures: [
        [
          "theme, mode",
          "localStorage",
          "Until cleared",
          "Colour scheme chosen with the theme switcher.",
        ],
        [
          "cultivars.consent",
          "localStorage",
          "Until cleared",
          "The choices you save in this panel.",
        ],
      ],
    },
    {
      id: "analytics",
      title: "Analytics",
      locked: false,
      description:
        "Google Analytics, used to see which pages are read and where the " +
        "site breaks. On by default; switch it off here and Google receives " +
        "only cookieless, non-identifying pings.",
      disclosures: [
        ["_ga", "Cookie (Google)", "2 years", "Distinguishes visitors."],
        [
          "_ga_*",
          "Cookie (Google)",
          "2 years",
          "Keeps session state for this property.",
        ],
      ],
    },
  ];

  /* ---- storage ---------------------------------------------------------- */

  function readState() {
    try {
      var raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      var parsed = JSON.parse(raw);
      if (!parsed || parsed.version !== VERSION) return null;
      return parsed;
    } catch (err) {
      return null;
    }
  }

  function writeState(choices) {
    var state = {
      version: VERSION,
      decidedAt: new Date().toISOString(),
      choices: choices,
    };
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (err) {
      /* Storage may be blocked; the choice still applies for this page. */
    }
    return state;
  }

  /* ---- consent effects --------------------------------------------------- */

  function gtagUpdate(granted) {
    if (typeof window.gtag !== "function") return;
    window.gtag("consent", "update", {
      analytics_storage: granted ? "granted" : "denied",
    });
  }

  function expireCookie(name, domain) {
    var base = name + "=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/";
    document.cookie = base;
    if (domain) document.cookie = base + "; domain=" + domain;
  }

  function deleteAnalyticsCookies() {
    var host = window.location.hostname;
    var parts = host.split(".");
    var parent = parts.length > 2 ? parts.slice(-2).join(".") : host;
    document.cookie.split(";").forEach(function (entry) {
      var name = entry.split("=")[0].trim();
      if (name === "_ga" || name.indexOf("_ga_") === 0) {
        expireCookie(name, null);
        expireCookie(name, host);
        expireCookie(name, "." + host);
        expireCookie(name, "." + parent);
      }
    });
  }

  function applyChoices(choices) {
    var analytics = Boolean(choices.analytics);
    gtagUpdate(analytics);
    if (!analytics) deleteAnalyticsCookies();
  }

  /* ---- markup ------------------------------------------------------------ */

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (key) {
      if (key === "text") node.textContent = attrs[key];
      else if (key === "html") node.innerHTML = attrs[key];
      else node.setAttribute(key, attrs[key]);
    });
    (children || []).forEach(function (child) {
      node.appendChild(child);
    });
    return node;
  }

  var COOKIE_ICON =
    '<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">' +
    '<path d="M12 2.5a9.5 9.5 0 1 0 9.5 9.5c0-.6-.06-1.2-.17-1.77' +
    'a3.6 3.6 0 0 1-4.35-3.13 3.6 3.6 0 0 1-3.2-4.43A9.6 9.6 0 0 0 12 2.5z"/>' +
    '<circle cx="8.5" cy="10" r="1.1" fill="currentColor" stroke="none"/>' +
    '<circle cx="10" cy="15.5" r="1.1" fill="currentColor" stroke="none"/>' +
    '<circle cx="15" cy="14" r="1.1" fill="currentColor" stroke="none"/>' +
    "</svg>";

  function buildDisclosures(rows) {
    var thead = el("thead", {}, [
      el(
        "tr",
        {},
        ["Name", "Type", "Retention", "Purpose"].map(function (h) {
          return el("th", { text: h });
        }),
      ),
    ]);
    var tbody = el(
      "tbody",
      {},
      rows.map(function (row) {
        return el(
          "tr",
          {},
          row.map(function (cell, i) {
            return el("td", {}, [
              i === 0
                ? el("code", { text: cell })
                : document.createTextNode(cell),
            ]);
          }),
        );
      }),
    );
    return el("details", { class: "cv-consent-disclosures" }, [
      el("summary", { text: "View disclosures" }),
      el("table", {}, [thead, tbody]),
    ]);
  }

  function buildCategory(category, current) {
    var inputId = "cv-consent-" + category.id;
    var input = el("input", {
      type: "checkbox",
      role: "switch",
      class: "cv-switch",
      id: inputId,
      "data-category": category.id,
    });
    input.checked = category.locked ? true : Boolean(current[category.id]);
    if (category.locked) input.disabled = true;
    return el("section", { class: "cv-consent-category" }, [
      el("div", { class: "cv-consent-category-head" }, [
        el("label", { for: inputId, text: category.title }),
        input,
      ]),
      el("p", { text: category.description }),
      buildDisclosures(category.disclosures),
    ]);
  }

  function build(current) {
    var button = el("button", {
      type: "button",
      class: "cv-consent-button",
      "aria-label": "Storage preferences",
      title: "Storage preferences",
      html: COOKIE_ICON,
    });
    var backdrop = el("div", { class: "cv-consent-backdrop" });
    var title = el("h2", {
      id: "cv-consent-title",
      text: "Storage preferences",
    });
    var close = el("button", {
      type: "button",
      class: "cv-consent-close",
      "aria-label": "Close",
      html: "&times;",
    });
    var body = el("div", { class: "cv-consent-body" }, [
      el("p", {
        class: "cv-consent-intro",
        text:
          "This site stores a little data in your browser. Essential storage " +
          "keeps it working; analytics is on unless you switch it off. " +
          "Change your mind any time from the button in the corner.",
      }),
    ]);
    CATEGORIES.forEach(function (category) {
      body.appendChild(buildCategory(category, current));
    });
    var accept = el("button", {
      type: "button",
      class: "cv-consent-accept",
      text: "Accept all",
    });
    var save = el("button", {
      type: "button",
      class: "cv-consent-save",
      text: "Save preferences",
    });
    var drawer = el(
      "aside",
      {
        class: "cv-consent-drawer",
        role: "dialog",
        "aria-modal": "true",
        "aria-labelledby": "cv-consent-title",
        "aria-hidden": "true",
      },
      [
        el("div", { class: "cv-consent-header" }, [title, close]),
        body,
        el("div", { class: "cv-consent-footer" }, [save, accept]),
      ],
    );
    document.body.appendChild(button);
    document.body.appendChild(backdrop);
    document.body.appendChild(drawer);
    return {
      button: button,
      backdrop: backdrop,
      drawer: drawer,
      close: close,
      accept: accept,
      save: save,
    };
  }

  /* ---- behaviour --------------------------------------------------------- */

  function main() {
    var state = readState();
    var current = state ? state.choices : DEFAULTS;
    applyChoices(current);

    var ui = build(current);
    var lastFocus = null;

    function open() {
      lastFocus = document.activeElement;
      document.documentElement.classList.add("cv-consent-open");
      ui.drawer.setAttribute("aria-hidden", "false");
      ui.close.focus();
    }

    function close() {
      document.documentElement.classList.remove("cv-consent-open");
      ui.drawer.setAttribute("aria-hidden", "true");
      var target =
        lastFocus && lastFocus !== document.body && document.contains(lastFocus)
          ? lastFocus
          : ui.button;
      target.focus();
    }

    function readSwitches() {
      var choices = {};
      CATEGORIES.forEach(function (category) {
        if (category.locked) return;
        var input = ui.drawer.querySelector(
          '[data-category="' + category.id + '"]',
        );
        choices[category.id] = Boolean(input && input.checked);
      });
      return choices;
    }

    function commit(choices) {
      CATEGORIES.forEach(function (category) {
        if (category.locked) return;
        var input = ui.drawer.querySelector(
          '[data-category="' + category.id + '"]',
        );
        if (input) input.checked = Boolean(choices[category.id]);
      });
      writeState(choices);
      applyChoices(choices);
      close();
    }

    ui.button.addEventListener("click", open);
    ui.close.addEventListener("click", close);
    ui.backdrop.addEventListener("click", close);
    ui.save.addEventListener("click", function () {
      commit(readSwitches());
    });
    ui.accept.addEventListener("click", function () {
      var all = {};
      CATEGORIES.forEach(function (category) {
        if (!category.locked) all[category.id] = true;
      });
      commit(all);
    });
    document.addEventListener("keydown", function (event) {
      if (
        event.key === "Escape" &&
        document.documentElement.classList.contains("cv-consent-open")
      )
        close();
    });

    /* First visit: no decision recorded, so ask. */
    if (!state) open();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main);
  } else {
    main();
  }
})();
