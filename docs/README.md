# CD Locator

API e pipeline modular para otimização de localização de Centros de Distribuição (CD), com upload, validação, forecast, matriz de custos, otimização MILP (fallback heurístico), cenários e relatórios.

## Executar localmente

### Com Docker

```bash
docker compose -f infra/docker-compose.yml up --build
```

### Sem Docker

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn app.main:app --app-dir backend --host 0.0.0.0 --port 8000
```

## Endpoints

Upload customers:

```bash
curl -X POST "http://localhost:8000/upload/customers" -H "X-User-Role: editor" -H "X-User-Scopes: cd:write,cd:run" -F "file=@example/customers_toy.csv"
```

Upload sites:

```bash
curl -X POST "http://localhost:8000/upload/sites" -H "X-User-Role: editor" -F "file=@example/sites_toy.csv"
```

Upload suppliers:

```bash
curl -X POST "http://localhost:8000/upload/suppliers" -H "X-User-Role: editor" -F "file=@example/suppliers_toy.csv"
```

Upload tax rules:

```bash
curl -X POST "http://localhost:8000/upload/tax_rules" -H "X-User-Role: editor" -F "file=@example/tax_rules_toy.json"
```

Rodar análise síncrona:

```bash
curl -X POST "http://localhost:8000/run/analysis?sync=true" -H "X-User-Role: editor" -H "X-User-Scopes: cd:write,cd:run" -H "Content-Type: application/json" -d '{
  "project_id":"proj_toy",
  "params": {"horizon_months":12, "max_sites":3, "objective_weights":{"cost":0.7,"time":0.3}, "seed":42},
  "consent_to_use_sensitive_data": true
}'
```

Status:

```bash
curl "http://localhost:8000/status/run_123"
```

Resultado:

```bash
curl "http://localhost:8000/results/run_123"
```

## Contratos e templates

- Contratos Pydantic em `backend/app/models/schemas.py`.
- Validação de CEP, lat/lon, duplicidade e autocompletar geocoding mock em `validator.py`.
- Templates podem ser exportados via função `export_template`.

## Interpretação dos outputs

- `open_sites`: sites selecionados com custo fixo e utilização.
- `allocations`: alocação cliente-site com qty/custo/tempo.
- `objective_breakdown`: decomposição (`transport`, `fixed`, `tax`, `inventory`, `time_penalty`).

## Solver e fallback

- MILP com PuLP/CBC.
- Se falhar por tempo/status, usa `fallback_greedy` e retorna `solver_status="failed; used_fallback"`.
- Gurobi/CPLEX são opcionais e podem ser configurados via variáveis de ambiente e integração futura.

## Segurança e governança

- Use `.env` com `APP_SECRET_KEY` e `DB_URL`.
- Em produção, habilitar TLS/HTTPS no proxy/API gateway.
- RBAC stub via header `X-User-Role` (`viewer`, `editor`, `admin`).
- Campos sensíveis devem ser criptografados em repouso (ex.: `cryptography`, KMS).

## Escala e performance

- Heurística alvo < 30 min para 10k clientes / 200 sites.
- MILP indicado para até ~2k clientes em < 15 min em máquina dev.
- Recomenda-se particionar forecast/cost matrix e usar Postgres+PostGIS em produção.


## Novas variáveis de ambiente (produção)

- `ROUTING_BACKEND`: `haversine|osrm`
- `OSRM_URL`: endpoint base do OSRM
- `SOLVER_BACKEND`: `pulp|gurobi|cplex` (hooks já preparados)
- `JOB_BACKEND`: `inline|celery|rq` (integração futura)
- `ENABLE_OIDC`: habilita fluxo de autenticação real (stub no momento)
- `REQUIRED_SCOPE_UPLOAD` e `REQUIRED_SCOPE_RUN`: escopos exigidos por endpoint
- `AUDIT_LOG_PATH`: caminho do JSONL de trilha de auditoria
