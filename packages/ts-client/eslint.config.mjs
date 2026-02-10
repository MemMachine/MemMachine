import tseslint from '@typescript-eslint/eslint-plugin'
import tsParser from '@typescript-eslint/parser'
import importPlugin from 'eslint-plugin-import'

export default [
  {
    files: ['**/*.ts'],
    languageOptions: {
      parser: tsParser
    },
    plugins: {
      '@typescript-eslint': tseslint,
      import: importPlugin
    },
    rules: {
      'import/order': ['error', { groups: ['builtin', 'external', 'internal'], 'newlines-between': 'always' }]
    }
  }
]
