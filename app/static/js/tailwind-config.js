tailwind.config = {
    theme: {
        extend: {
            colors: {
                border: "hsl(var(--border))",
                input: "hsl(var(--input))",
                ring: "hsl(var(--ring))",
                background: "hsl(var(--background))",
                foreground: "hsl(var(--foreground))",
                primary: {
                    DEFAULT: "#0ea5e9", // Sky-500
                    foreground: "#ffffff",
                },
                secondary: {
                    DEFAULT: "#f1f5f9", // Slate-100
                    foreground: "#0f172a", // Slate-900
                },
                destructive: {
                    DEFAULT: "#ef4444", // Red-500
                    foreground: "#ffffff",
                },
                muted: {
                    DEFAULT: "#f8fafc", // Slate-50
                    foreground: "#64748b", // Slate-500
                },
                accent: {
                    DEFAULT: "#e0f2fe", // Sky-100
                    foreground: "#0369a1", // Sky-700
                },
            },
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
            }
        }
    }
}
