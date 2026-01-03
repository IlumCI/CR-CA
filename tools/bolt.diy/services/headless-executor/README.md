# Headless Executor Worker Service

Executes mandates using Playwright-controlled headless browsers with WebContainer.

## Features

- **Headless Browser Execution**: Uses Playwright to control Chromium
- **WebContainer Support**: Executes mandates in browser-based WebContainer
- **Parallel Execution**: Supports multiple concurrent executions
- **Auto-registration**: Automatically registers with Execution Governor
- **Health Monitoring**: Sends heartbeats to governor
- **Event Reporting**: Reports execution progress and events

## Configuration

Environment variables:

- `GOVERNOR_URL`: Execution Governor URL (default: http://localhost:3000)
- `BOLT_DIY_URL`: Bolt.diy application URL (default: http://localhost:5173)
- `WORKER_ID`: Unique worker identifier (default: auto-generated)
- `MAX_CONCURRENT`: Max concurrent executions (default: 1)
- `POLL_INTERVAL`: Polling interval in ms (default: 5000)
- `HEADLESS`: Run browser in headless mode (default: true)
- `EXECUTION_TIMEOUT`: Execution timeout in ms (default: 3600000)

## Development

```bash
# Install dependencies
pnpm install

# Install Playwright browsers
npx playwright install chromium

# Run in development mode
pnpm run dev

# Build
pnpm run build

# Run production
pnpm start
```

## Docker

```bash
# Build
docker build -t headless-executor .

# Run
docker run -e GOVERNOR_URL=http://governor:3000 \
           -e BOLT_DIY_URL=http://bolt:5173 \
           -e MAX_CONCURRENT=2 \
           headless-executor
```

## Architecture

1. Worker registers with Execution Governor
2. Worker polls governor for mandates (or receives via WebSocket)
3. Worker launches Playwright browser
4. Worker navigates to bolt.diy execution page
5. Execution page auto-executes mandate
6. Worker captures events and reports to governor
7. Worker reports completion to governor

