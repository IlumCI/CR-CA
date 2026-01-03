FROM node:22-bookworm-slim AS build

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Use pnpm
RUN corepack enable && corepack prepare pnpm@9.15.9 --activate

# Copy package files
COPY package.json ./
COPY pnpm-lock.yaml* ./
# Install dependencies (use --no-frozen-lockfile if lockfile doesn't exist)
RUN if [ -f pnpm-lock.yaml ]; then pnpm install --frozen-lockfile; else pnpm install --no-frozen-lockfile; fi

# Copy source
COPY . .

# Build TypeScript
RUN pnpm run build

# Production stage with Playwright
FROM mcr.microsoft.com/playwright:v1.41.0-jammy AS production

WORKDIR /app

ENV NODE_ENV=production

# Use pnpm
RUN corepack enable && corepack prepare pnpm@9.15.9 --activate

# Copy built files
COPY --from=build /app/dist ./dist
COPY --from=build /app/package.json ./
COPY --from=build /app/node_modules ./node_modules

# Playwright browsers are already installed in base image

CMD ["node", "dist/index.js"]

