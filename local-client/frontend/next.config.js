/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Keep a production build separate from the active development server.
  distDir: process.env.NODE_ENV === "production" ? ".next-production" : ".next",
  images: {
    // Allow picsum.photos for mock frame images.
    // Replace / extend with your real frame server domain when ingestion is ready.
    remotePatterns: [
      { protocol: "https", hostname: "picsum.photos" },
    ],
  },
};

module.exports = nextConfig;
