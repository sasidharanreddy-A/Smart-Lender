/* Smart Lender v2 — ULTRA SMOOTH EDITION
   Performance features:
   - Debounced scroll handler (no layout thrashing)
   - Passive event listeners
   - requestAnimationFrame batching for DOM writes
   - Reduced particle count on mobile
   - Clean IntersectionObserver usage
   - will-change managed via CSS (never in JS)
   - Optimized counter animation (no textContent thrash)
*/
(function () {
  "use strict";

  /* =====================================================================
     Utility: Debounce — limits how often a function can fire
     ===================================================================== */
  function debounce(fn, delay) {
    let timer = null;
    return function () {
      const context = this;
      const args = arguments;
      if (timer) clearTimeout(timer);
      timer = setTimeout(function () {
        timer = null;
        fn.apply(context, args);
      }, delay);
    };
  }

  /* =====================================================================
     1. Scroll-triggered Reveal Animations (Intersection Observer)
        Uses passive observers — no main-thread blocking
     ===================================================================== */
  function initScrollReveal() {
    var elements = document.querySelectorAll(
      ".reveal, .reveal-left, .reveal-right, .reveal-scale"
    );

    if (!elements.length) return;

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add("revealed");
            observer.unobserve(entry.target);
          }
        });
      },
      {
        threshold: 0.05,
        rootMargin: "0px 0px -30px 0px",
      }
    );

    // Batch DOM reads — use forEach on the NodeList
    Array.prototype.forEach.call(elements, function (el) {
      observer.observe(el);
    });
  }

  /* =====================================================================
     2. Back-to-Top Button with requestAnimationFrame scroll check
     ===================================================================== */
  function initBackToTop() {
    var btn = document.createElement("button");
    btn.className = "back-to-top";
    btn.innerHTML = '<i class="bi bi-arrow-up"></i>';
    document.body.appendChild(btn);

    var ticking = false;
    var scrollHandler = function () {
      if (!ticking) {
        window.requestAnimationFrame(function () {
          if (window.scrollY > 400) {
            if (!btn.classList.contains("visible")) {
              btn.classList.add("visible");
            }
          } else {
            if (btn.classList.contains("visible")) {
              btn.classList.remove("visible");
            }
          }
          ticking = false;
        });
        ticking = true;
      }
    };

    // Use passive: true — never blocks scrolling
    window.addEventListener("scroll", scrollHandler, { passive: true });

    btn.addEventListener("click", function () {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  /* =====================================================================
     3. Floating Particles — lightweight, reduced on mobile
     ===================================================================== */
  function initParticles() {
    // Fewer particles on mobile for performance
    var isMobile = window.innerWidth < 768;
    var count = isMobile ? 6 : 12;

    var container = document.createElement("div");
    container.className = "particles-container";

    for (var i = 0; i < count; i++) {
      var particle = document.createElement("div");
      particle.className = "particle";
      container.appendChild(particle);
    }

    document.body.prepend(container);
  }

  /* =====================================================================
     4. Form Enhancements — optimized with fewer reflows
     ===================================================================== */
  function initFormEnhancements() {
    var form = document.getElementById("predictForm");
    if (!form) return;

    // Cache frequently accessed elements
    var loanAmount = form.querySelector('[name="LoanAmount"]');
    var term = form.querySelector('[name="Loan_Amount_Term"]');
    var submitBtn = form.querySelector('[type="submit"]');

    // --- Auto-fill defaults and smart hints ---
    if (loanAmount && term) {
      loanAmount.addEventListener("change", function () {
        var v = parseFloat(this.value);
        if (!isNaN(v) && v > 0 && !term.dataset.touched) {
          if (v > 500) {
            term.value = "480";
          } else if (v > 300) {
            term.value = "360";
          } else if (v > 100) {
            term.value = "240";
          }
        }
      });
      term.addEventListener("change", function () {
        term.dataset.touched = "1";
      });
    }

    // --- Real-time validation feedback ---
    var numericFields = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"];
    numericFields.forEach(function (name) {
      var el = form.querySelector('[name="' + name + '"]');
      if (!el) return;
      el.addEventListener("input", function () {
        var v = parseFloat(this.value);
        if (isNaN(v) || v < 0) {
          this.classList.add("is-invalid");
          this.classList.remove("is-valid");
        } else {
          this.classList.remove("is-invalid");
          this.classList.add("is-valid");
        }
      });
    });

    // --- Credit History validation ---
    var creditHistory = form.querySelector('[name="Credit_History"]');
    if (creditHistory) {
      creditHistory.addEventListener("change", function () {
        var v = this.value;
        if (v !== "0" && v !== "1") {
          this.classList.add("is-invalid");
          this.classList.remove("is-valid");
        } else {
          this.classList.remove("is-invalid");
          this.classList.add("is-valid");
        }
      });
    }

    // --- Animate submit button on click ---
    if (submitBtn) {
      form.addEventListener("submit", function () {
        submitBtn.disabled = true;
        submitBtn.innerHTML =
          '<div class="spinner-border spinner-border-sm me-2" role="status"></div> Analyzing...';
      });
    }

    // --- Form submission validation ---
    form.addEventListener("submit", function (event) {
      var valid = true;

      // Batch DOM reads: get all values first
      form.querySelectorAll(".is-invalid").forEach(function (el) {
        el.classList.remove("is-invalid");
      });

      numericFields.forEach(function (name) {
        var el = form.querySelector('[name="' + name + '"]');
        if (!el) return;
        var v = parseFloat(el.value);
        if (isNaN(v) || v < 0 || el.value.trim() === "") {
          el.classList.add("is-invalid");
          valid = false;
        }
      });

      var termEl = form.querySelector('[name="Loan_Amount_Term"]');
      if (termEl) {
        var termV = parseFloat(termEl.value);
        if (isNaN(termV) || termV <= 0 || termEl.value.trim() === "") {
          termEl.classList.add("is-invalid");
          valid = false;
        }
      }

      var creditEl = form.querySelector('[name="Credit_History"]');
      if (creditEl) {
        var creditV = creditEl.value;
        if (creditV !== "0" && creditV !== "1") {
          creditEl.classList.add("is-invalid");
          valid = false;
        }
      }

      if (!valid) {
        event.preventDefault();
        if (submitBtn) {
          submitBtn.disabled = false;
          submitBtn.innerHTML =
            '<i class="bi bi-stars"></i> Predict Loan Approval';
        }
        var first = form.querySelector(".is-invalid");
        if (first) {
          first.focus();
          first.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }
    });

    // --- Blur validation styling ---
    form.querySelectorAll(".form-control, .form-select").forEach(function (el) {
      el.addEventListener("blur", function () {
        if (this.value && !this.classList.contains("is-invalid")) {
          this.classList.add("is-valid");
        }
      });
    });
  }

  /* =====================================================================
     5. Counter Animation — GPU-friendly, uses rAF batching
     ===================================================================== */
  function initCounterAnimations() {
    var counters = document.querySelectorAll(".metric-value");
    if (!counters.length) return;

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            var el = entry.target;
            var text = el.textContent.trim();
            var num = parseFloat(text);
            if (!isNaN(num)) {
              animateCounter(el, num);
            }
            observer.unobserve(el);
          }
        });
      },
      { threshold: 0.3 }
    );

    Array.prototype.forEach.call(counters, function (el) {
      observer.observe(el);
    });
  }

  function animateCounter(el, target) {
    var original = el.textContent.trim();
    var suffix = original.replace(/[\d.,\s]/g, "");
    var duration = 800; // slightly faster for smoother feel
    var startTime = performance.now();
    var isInteger = Number.isInteger(target);

    function update(currentTime) {
      var elapsed = currentTime - startTime;
      var progress = Math.min(elapsed / duration, 1);
      // Ease-out cubic
      var eased = 1 - Math.pow(1 - progress, 3);
      var current = eased * target;

      // Batch: only write once per frame
      if (isInteger) {
        el.textContent = Math.round(current) + suffix;
      } else {
        el.textContent = current.toFixed(1) + suffix;
      }

      if (progress < 1) {
        window.requestAnimationFrame(update);
      } else {
        // Restore original text for perfect match
        el.textContent = original;
      }
    }

    window.requestAnimationFrame(update);
  }

  /* =====================================================================
     6. Interactive Tooltips — lightweight, lazy
     ===================================================================== */
  function initTooltips() {
    var elements = document.querySelectorAll("[data-tooltip]");
    if (!elements.length) return;

    Array.prototype.forEach.call(elements, function (el) {
      var text = el.dataset.tooltip;
      if (!text) return;

      var tooltip = document.createElement("div");
      tooltip.className = "tooltip-glass";
      tooltip.textContent = text;
      tooltip.style.cssText =
        "position:absolute;z-index:1000;pointer-events:none;" +
        "opacity:0;transition:opacity 0.15s ease;";
      el.style.position = "relative";
      el.appendChild(tooltip);

      el.addEventListener("mouseenter", function () {
        tooltip.style.opacity = "1";
      });
      el.addEventListener("mouseleave", function () {
        tooltip.style.opacity = "0";
      });
    });
  }

  /* =====================================================================
     7. Track window resize to update particle count
        (called once, not spammed)
     ===================================================================== */
  var resizeTimeout = null;
  function initResponsiveParticles() {
    window.addEventListener(
      "resize",
      function () {
        // Debounce: wait 300ms after resize ends
        if (resizeTimeout) clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(function () {
          // Check if particles need adjusting
          var container = document.querySelector(".particles-container");
          if (!container) return;
          var isMobile = window.innerWidth < 768;
          var currentCount = container.children.length;
          var targetCount = isMobile ? 6 : 12;

          if (currentCount !== targetCount) {
            // Replace particles container
            container.innerHTML = "";
            for (var i = 0; i < targetCount; i++) {
              var particle = document.createElement("div");
              particle.className = "particle";
              container.appendChild(particle);
            }
          }
        }, 300);
      },
      { passive: true }
    );
  }

  /* =====================================================================
     Initialize everything on DOM ready
     — Order: particles first (background) → reveal → UI → forms
     ===================================================================== */
  document.addEventListener("DOMContentLoaded", function () {
    // Background effects first (lower priority, no layout impact)
    if (window.requestIdleCallback) {
      window.requestIdleCallback(function () {
        initParticles();
        initResponsiveParticles();
      });
    } else {
      // Fallback for browsers without requestIdleCallback
      setTimeout(function () {
        initParticles();
        initResponsiveParticles();
      }, 50);
    }

    // Critical UI elements
    initScrollReveal();
    initBackToTop();
    initFormEnhancements();
    initCounterAnimations();
    initTooltips();
  });
})();
