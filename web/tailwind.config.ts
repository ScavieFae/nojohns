import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          900: "#0a0a0f",
          800: "#12121a",
          700: "#1a1a25",
          600: "#222230",
        },
        accent: {
          green: "#22c55e",
          red: "#ef4444",
          blue: "#3b82f6",
          yellow: "#eab308",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
} satisfies Config;
