# Execution Governor Service

Manages autonomous mandate execution with concurrency control, priority queuing, rate limiting, and resource management.

## Features

- **Priority Queue**: Higher priority mandates execute first
- **Concurrency Control**: Configurable max parallel executions
- **Rate Limiting**: Max mandates per time window
- **Budget Enforcement**: Per-mandate and global budget limits
- **Resource Management**: CPU/memory limits per execution
- **Worker Pool Management**: Register/unregister workers, health checks
- **Fault Tolerance**: Retry failed mandates, worker failover
- **Real-time Events**: WebSocket broadcasting for observability

## Configuration

Environment variables:

- `PORT`: Service port (default: 3000)
- `MAX_CONCURRENT_EXECUTIONS`: Max parallel executions (default: 5)
- `MAX_MANDATES_PER_MINUTE`: Rate limit (default: 20)
- `GLOBAL_BUDGET_LIMIT`: Global budget in dollars (default: 1000.0)
- `CPU_LIMIT`: CPU limit per execution (default: 2.0)
- `MEMORY_LIMIT`: Memory limit in MB (default: 4096)
- `WORKER_TIMEOUT`: Worker timeout in ms (default: 3600000)
- `RETRY_ATTEMPTS`: Max retry attempts (default: 3)
- `WORKER_HEALTH_CHECK_INTERVAL`: Health check interval in ms (default: 30000)
- `QUEUE_CHECK_INTERVAL`: Queue processing interval in ms (default: 1000)

## API Endpoints

- `POST /mandates` - Accept mandate from bolt.diy API
- `GET /mandates/:id/status` - Get mandate execution status
- `GET /mandates/:id` - Get full mandate details
- `POST /workers/register` - Register headless worker
- `POST /workers/:id/heartbeat` - Worker heartbeat
- `POST /workers/:id/request-mandate` - Worker requests next mandate
- `POST /workers/:id/report-progress` - Worker reports execution progress
- `POST /workers/:id/complete` - Worker reports mandate completion
- `GET /governor/state` - Get governor state
- `GET /governor/metrics` - Get execution metrics

## WebSocket

Connect to `ws://localhost:3000` and send:

- `{"type": "subscribe", "mandate_id": "..."}` - Subscribe to mandate events
- `{"type": "unsubscribe", "mandate_id": "..."}` - Unsubscribe from mandate events

## Development

```bash
# Install dependencies
pnpm install

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
docker build -t execution-governor .

# Run
docker run -p 3000:3000 execution-governor
```

