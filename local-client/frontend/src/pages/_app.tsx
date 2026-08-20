import type { AppProps } from "next/app";
import Head from "next/head";
import { Space_Mono } from "next/font/google";
import "@/styles/globals.css";

// Retro display font — only ever applied to English UI chrome (titles, tab
// labels, buttons) via the `font-retro` Tailwind utility, never to Vietnamese
// content, since this subset doesn't cover Vietnamese diacritics.
const spaceMono = Space_Mono({
  subsets: ["latin"],
  weight: ["400", "700"],
  variable: "--font-retro",
});

export default function App({ Component, pageProps }: AppProps) {
  return (
    <div className={spaceMono.variable}>
      <Head>
        <link rel="icon" href="/favicon.jpg" type="image/jpeg" />
      </Head>
      <Component {...pageProps} />
    </div>
  );
}
