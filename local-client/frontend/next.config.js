/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    // Allow picsum.photos for mock frame images.
    // Replace / extend with your real frame server domain when ingestion is ready.
    remotePatterns: [
      { protocol: "https", hostname: "picsum.photos" },
    ],
  },
};

module.exports = nextConfig;
