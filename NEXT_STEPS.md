# NEXT STEPS (produção)

## Implementado nesta iteração (fundação)

1. **Roteamento real (OSRM) com cache**: hook `route_via_osrm` com `lru_cache` e fallback seguro para haversine.
2. **Feature flag de solver comercial**: `SOLVER_BACKEND` e ponto de extensão para Gurobi/CPLEX no optimizer.
3. **Autorização por escopo (stub)**: header `X-User-Scopes` com validações para upload e execução.
4. **Governança/auditoria**: trilha de auditoria JSONL em `data/audit/events.jsonl` para eventos de criação/conclusão de run.
5. **Base para scheduler**: `JOB_BACKEND` exposto na resposta async para integração com Celery/RQ.
6. **Persistência de estado**: `projects`/`runs` persistidos em SQLite (camada `StateStore`) para transição futura a Postgres.
7. **Hook GraphHopper**: integração com fallback seguro no engine de custos.
8. **Cache persistente O-D**: tabela `od_cache` no `StateStore` com leitura/gravação no cost engine.
9. **Retries básicos de execução**: wrapper `_run_pipeline_with_retries` pronto para workers distribuídos.

## Próximos incrementos

1. Integrar GraphHopper e cache persistente de matriz O-D (Redis/Postgres).
2. Migrar estado de `PROJECTS`/`RUNS` para Postgres + PostGIS.
3. Implementar execução real com Gurobi/CPLEX (licença e parâmetros de tuning).
4. Integrar OIDC/SSO real e RBAC por escopos persistidos.
5. Adicionar worker distribuído real (Celery/RQ) com DLQ persistente.
6. Versionar datasets/modelos com catálogo central e assinatura criptográfica gerenciada (KMS).
