import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SMS Spam Classifier",
  description:
    "Detect whether an SMS message is spam or ham using a pre-trained CountVectorizer + Multinomial Naive Bayes model.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
