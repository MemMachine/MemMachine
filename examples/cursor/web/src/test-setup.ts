import { vi } from 'vitest';

// Mock TextEncoder and TextDecoder for Node.js environment
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

// Mock atob and btoa for Node.js environment
global.atob = vi.fn((str: string) => {
  return Buffer.from(str, 'base64').toString('binary');
});

global.btoa = vi.fn((str: string) => {
  return Buffer.from(str, 'binary').toString('base64');
});
