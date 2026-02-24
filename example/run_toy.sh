#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

docker compose -f infra/docker-compose.yml up --build -d
sleep 5

curl -s -X POST "http://localhost:8000/upload/customers" -H "X-User-Role: editor" -F "file=@example/customers_toy.csv" > example/output/upload_customers.json
curl -s -X POST "http://localhost:8000/upload/sites" -H "X-User-Role: editor" -F "file=@example/sites_toy.csv" > example/output/upload_sites.json
curl -s -X POST "http://localhost:8000/upload/suppliers" -H "X-User-Role: editor" -F "file=@example/suppliers_toy.csv" > example/output/upload_suppliers.json
curl -s -X POST "http://localhost:8000/upload/tax_rules" -H "X-User-Role: editor" -F "file=@example/tax_rules_toy.json" > example/output/upload_tax.json

curl -s -X POST "http://localhost:8000/run/analysis?sync=true" \
  -H "X-User-Role: editor" \
  -H "Content-Type: application/json" \
  -d '{"project_id":"proj_toy","params":{"horizon_months":12,"max_sites":3,"objective_weights":{"cost":0.7,"time":0.3},"seed":42},"consent_to_use_sensitive_data":true}' \
  > example/output/run_response.json

echo "Outputs in example/output"
