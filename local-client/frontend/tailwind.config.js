/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        retro: ["var(--font-retro)", "ui-monospace", "monospace"],
      },
      colors: {
        cream: {
          DEFAULT: "#f2ead9", // page background — matches globals.css body
          card: "#f8f2e2",    // slightly lighter — cards/inputs/panels
        },
      },
    },
  },
  plugins: [],
};
