/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      borderColor: {
        DEFAULT: "hsl(var(--border))",
      },
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        gray: "hsl(var(--gray))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          gray: "hsl(var(--primary-gray))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          gray: "hsl(var(--secondary-gray))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          gray: "hsl(var(--accent-gray))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          gray: "hsl(var(--muted-gray))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          gray: "hsl(var(--card-gray))",
        },
      },
    },
  },
  plugins: [],
};
