-- Migration 001 rollback: remove HR serving-store schemas and roles
-- connect-kb-hr#1
--
-- WARNING: Destructive. Only run to fully tear down the HR serving store.
-- Active releases, chunks, and usage records will be permanently lost.

DROP SCHEMA IF EXISTS hr_employer CASCADE;
DROP SCHEMA IF EXISTS hr_employee CASCADE;

-- Roles: only drop if no other objects depend on them
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hr_employer_reader') THEN
    DROP ROLE hr_employer_reader;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hr_employee_reader') THEN
    DROP ROLE hr_employee_reader;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hr_publisher') THEN
    DROP ROLE hr_publisher;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hr_policy_runtime') THEN
    DROP ROLE hr_policy_runtime;
  END IF;
END $$;
