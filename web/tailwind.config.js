/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg:        '#0a0a0f',
        surface:   '#12121a',
        border:    '#1e1e2e',
        primary:   '#6366f1',
        secondary: '#22d3ee',
        success:   '#4ade80',
        warning:   '#fb923c',
        danger:    '#f87171',
        muted:     '#94a3b8',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      boxShadow: {
        glow:    '0 0 20px rgba(99,102,241,0.4)',
        'glow-cyan': '0 0 20px rgba(34,211,238,0.4)',
        'glow-green': '0 0 20px rgba(74,222,128,0.3)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'draw': 'draw 2s ease forwards',
      },
    },
  },
  plugins: [],
}
