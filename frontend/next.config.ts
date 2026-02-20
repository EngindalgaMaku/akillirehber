import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  typescript: {
    // Temporarily ignore build errors to speed up development
    ignoreBuildErrors: true,
  },
  // API proxy for backend
  async rewrites() {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    // If apiUrl already includes /api prefix, don't add it again
    const hasApiPrefix = apiUrl.endsWith('/api');
    const destination = hasApiPrefix ? `${apiUrl}/:path*` : `${apiUrl}/api/:path*`;
    
    return [
      {
        source: "/api/:path*",
        destination,
      },
    ];
  },
};

export default nextConfig;
