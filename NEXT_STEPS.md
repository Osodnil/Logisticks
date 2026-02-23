# NEXT STEPS (produção)

1. Integrar roteamento real via OSRM/GraphHopper com cache de matriz O-D.
2. Migrar armazenamento para Postgres + PostGIS para consultas geográficas eficientes.
3. Suportar solver comercial (Gurobi/CPLEX) por feature flag/env.
4. Implementar autenticação real (OIDC/SSO) e autorização por escopos.
5. Adicionar job scheduler (Celery/RQ) com retries, DLQ e observabilidade.
6. Melhorar governança: versionamento de datasets/modelos e trilha de auditoria.
