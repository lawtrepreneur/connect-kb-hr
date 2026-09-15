-- Migration 001 rollback: remove unified corpus/policy schemas and roles
-- connect-kb-hr — destroys kb and hr_policy schemas entirely
--
-- WARNING: Destructive. Only run to fully tear down the unified corpus.
-- Active releases, chunks, and usage records will be permanently lost.

DROP SCHEMA IF EXISTS hr_policy CASCADE;
DROP SCHEMA IF EXISTS kb CASCADE;
DROP ROLE IF EXISTS hr_runtime;
DROP ROLE IF EXISTS hr_policy_writer;
DROP ROLE IF EXISTS hr_publisher;
DROP ROLE IF EXISTS hr_migrator;
