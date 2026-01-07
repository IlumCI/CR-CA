#!/usr/bin/env node
/**
 * Windows-compatible bindings generator for Wrangler.
 * Reads .env.local and generates --binding flags for environment variables.
 */

const fs = require('fs');
const path = require('path');

/**
 * Extracts environment variable names from the TypeScript interface.
 * @returns {string[]} Array of environment variable names
 */
function extractEnvVars() {
  try {
    const workerConfigPath = path.join(__dirname, 'worker-configuration.d.ts');
    const content = fs.readFileSync(workerConfigPath, 'utf-8');
    const matches = content.match(/[A-Z_]+:/g) || [];
    return matches.map((match) => match.replace(':', ''));
  } catch {
    return [];
  }
}

/**
 * Generates Wrangler bindings from .env.local or environment variables.
 * @returns {string} Space-separated --binding flags
 */
function generateBindings() {
  let bindings = [];
  const envLocalPath = path.join(__dirname, '.env.local');

  // First try to read from .env.local if it exists
  if (fs.existsSync(envLocalPath)) {
    const content = fs.readFileSync(envLocalPath, 'utf-8');
    const lines = content.split(/\r?\n/);

    for (const line of lines) {
      const trimmed = line.trim();
      // Skip comments and empty lines
      if (!trimmed || trimmed.startsWith('#')) {
        continue;
      }

      const equalIndex = trimmed.indexOf('=');
      if (equalIndex === -1) {
        continue;
      }

      const name = trimmed.substring(0, equalIndex).trim();
      let value = trimmed.substring(equalIndex + 1).trim();

      // Remove surrounding quotes if present
      if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }

      if (name && value) {
        bindings.push(`--binding ${name}=${value}`);
      }
    }
  } else {
    // If .env.local doesn't exist, use environment variables defined in .d.ts
    const envVars = extractEnvVars();
    for (const varName of envVars) {
      const value = process.env[varName];
      if (value) {
        bindings.push(`--binding ${varName}=${value}`);
      }
    }
  }

  return bindings.join(' ');
}

// Output bindings
console.log(generateBindings());

