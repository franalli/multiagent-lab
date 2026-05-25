// frontend/convexClient.ts
//
// The single ConvexReactClient + provider wiring for the Next.js app.
//
// During the trial demo, the architectural point is: components don't poll
// or set up sockets. They useQuery(...) against this client; the moment the
// agent commits a mutation from inside the sandbox, every subscribed
// component re-renders. The websocket plumbing is the Convex client; we
// never touch it.

import { ConvexReactClient } from "convex/react";

// NEXT_PUBLIC_CONVEX_URL is set at build time by `npx convex dev` writing
// .env.local. The exuberant-albatross-781 default is the dev deployment
// the POC was bootstrapped against.
const convexUrl =
  process.env.NEXT_PUBLIC_CONVEX_URL ??
  "https://exuberant-albatross-781.convex.cloud";

export const convex = new ConvexReactClient(convexUrl);
