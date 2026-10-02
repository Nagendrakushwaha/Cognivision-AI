#!/usr/bin/env node

/**
 * Cognivision AI - Unified Development Runner
 * Automatically starts both:
 *   1. FastAPI Python Backend on http://127.0.0.1:8001
 *   2. React + Vite Frontend on http://localhost:5173
 *
 * Ensures:
 *   - Automatic detection of Python 3.13 / Python environment
 *   - Stale port cleanup before starting (avoids ECONNREFUSED & EADDRINUSE)
 *   - Synchronized startup (backend health check before frontend starts)
 *   - Clean shutdown of all child processes on Ctrl+C
 */

const { spawn, spawnSync, execSync } = require('child_process');
const http = require('http');
const path = require('path');
const fs = require('fs');

const ROOT_DIR = path.resolve(__dirname, '..');
const FRONTEND_DIR = path.resolve(ROOT_DIR, 'frontend');
const VITE_BIN = path.resolve(FRONTEND_DIR, 'node_modules', 'vite', 'bin', 'vite.js');

// ANSI Terminal Colors
const cyan = '\x1b[36m';
const magenta = '\x1b[35m';
const green = '\x1b[32m';
const yellow = '\x1b[33m';
const red = '\x1b[31m';
const gray = '\x1b[90m';
const bold = '\x1b[1m';
const reset = '\x1b[0m';

let backendProcess = null;
let frontendProcess = null;
let isShuttingDown = false;

function printBanner() {
  console.log(`\n${cyan}${bold}============================================================${reset}`);
  console.log(`${cyan}${bold}               COGNIVISION AI - DEV SERVER                  ${reset}`);
  console.log(`${cyan}${bold}============================================================${reset}`);
  console.log(`${gray}Starting unified backend (FastAPI :8001) + frontend (Vite :5173)...${reset}\n`);
}

function detectPython() {
  const candidates = [
    { cmd: 'py', args: ['-3.13'] },
    { cmd: 'python', args: [] },
    { cmd: 'py', args: [] },
    { cmd: 'python3', args: [] }
  ];

  for (const candidate of candidates) {
    try {
      const res = spawnSync(candidate.cmd, [...candidate.args, '--version'], {
        encoding: 'utf-8',
        windowsHide: true
      });
      if (res.status === 0) {
        const ver = (res.stdout || res.stderr || '').trim();
        return { ...candidate, version: ver };
      }
    } catch {
      // try next candidate
    }
  }
  return null;
}

function freePortIfHeld(port) {
  if (process.platform === 'win32') {
    try {
      const output = execSync(`netstat -ano | findstr :${port}`, {
        encoding: 'utf-8',
        stdio: ['ignore', 'pipe', 'ignore']
      });
      const lines = output.trim().split(/\r?\n/);
      for (const line of lines) {
        const parts = line.trim().split(/\s+/);
        if (parts.length >= 5 && parts[1].includes(`:${port}`) && parts[3] === 'LISTENING') {
          const pid = parts[4];
          if (pid && pid !== '0' && pid !== String(process.pid)) {
            try {
              execSync(`taskkill /pid ${pid} /F`, { stdio: 'ignore' });
              console.log(`${yellow}[Init] Cleared stale process (PID ${pid}) from port ${port}${reset}`);
            } catch {}
          }
        }
      }
    } catch {}
  }
}

function checkBackendHealth(timeoutMs = 1000) {
  return new Promise((resolve) => {
    const req = http.get('http://127.0.0.1:8001/api/health', { timeout: timeoutMs }, (res) => {
      resolve(res.statusCode === 200);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function waitForBackend(maxAttempts = 30) {
  for (let i = 0; i < maxAttempts; i++) {
    const ok = await checkBackendHealth(1200);
    if (ok) return true;
    await new Promise((r) => setTimeout(r, 600));
  }
  return false;
}

function openBrowser(url) {
  try {
    if (process.platform === 'win32') {
      spawn('cmd.exe', ['/c', 'start', '', url], { detached: true, stdio: 'ignore' });
    } else if (process.platform === 'darwin') {
      spawn('open', [url], { detached: true, stdio: 'ignore' });
    } else {
      spawn('xdg-open', [url], { detached: true, stdio: 'ignore' });
    }
  } catch {}
}

function cleanShutdown() {
  if (isShuttingDown) return;
  isShuttingDown = true;
  console.log(`\n${yellow}[Cognivision AI] Shutting down servers cleanly...${reset}`);

  if (backendProcess && backendProcess.pid) {
    try {
      if (process.platform === 'win32') {
        execSync(`taskkill /pid ${backendProcess.pid} /T /F`, { stdio: 'ignore' });
      } else {
        process.kill(-backendProcess.pid, 'SIGINT');
      }
    } catch {}
  }

  if (frontendProcess && frontendProcess.pid) {
    try {
      if (process.platform === 'win32') {
        execSync(`taskkill /pid ${frontendProcess.pid} /T /F`, { stdio: 'ignore' });
      } else {
        process.kill(-frontendProcess.pid, 'SIGINT');
      }
    } catch {}
  }

  console.log(`${green}[Cognivision AI] All servers stopped.${reset}`);
  process.exit(0);
}

process.on('SIGINT', cleanShutdown);
process.on('SIGTERM', cleanShutdown);
process.on('SIGHUP', cleanShutdown);

async function main() {
  printBanner();

  // 1. Python Environment Check
  const py = detectPython();
  if (!py) {
    console.error(`${red}[Error] Python is not installed or not in PATH!${reset}`);
    console.error('Please install Python 3.13 or add Python to your system PATH.');
    process.exit(1);
  }
  console.log(`${green}✔ Python Detected:${reset} ${py.version} (${[py.cmd, ...py.args].join(' ')})`);

  // 2. Clear stale Vite dev server port if held by dead process
  freePortIfHeld(5173);

  // 3. Check if Backend is already alive
  const alreadyRunning = await checkBackendHealth(800);
  if (alreadyRunning) {
    console.log(`${green}✔ FastAPI Backend is already running on http://127.0.0.1:8001${reset}`);
  } else {
    // Clear stale socket on port 8001 if any
    freePortIfHeld(8001);

    console.log(`${cyan}Starting FastAPI Backend on http://127.0.0.1:8001 ...${reset}`);
    backendProcess = spawn(
      py.cmd,
      [...py.args, '-m', 'uvicorn', 'backend.app.main:app', '--host', '127.0.0.1', '--port', '8001', '--reload'],
      {
        cwd: ROOT_DIR,
        stdio: ['ignore', 'pipe', 'pipe']
      }
    );

    backendProcess.stdout.on('data', (d) => {
      const msg = d.toString();
      // Print formatted output with backend tag
      msg.split(/\r?\n/).forEach((line) => {
        if (line.trim()) console.log(`${cyan}[backend]${reset} ${line}`);
      });
    });

    backendProcess.stderr.on('data', (d) => {
      const msg = d.toString();
      msg.split(/\r?\n/).forEach((line) => {
        if (line.trim()) console.log(`${cyan}[backend]${reset} ${line}`);
      });
    });

    backendProcess.on('exit', (code) => {
      if (!isShuttingDown) {
        console.log(`${red}[backend] Process exited with code ${code}${reset}`);
      }
    });

    // Wait for backend to be healthy
    process.stdout.write(`${gray}Waiting for FastAPI backend to initialize...${reset}`);
    const ready = await waitForBackend();
    if (ready) {
      console.log(`\r${green}✔ FastAPI Backend is healthy on http://127.0.0.1:8001${reset}             `);
    } else {
      console.log(`\r${yellow}⚠ Backend is taking longer than usual to respond, proceeding with frontend...${reset}`);
    }
  }

  // 4. Start Vite Frontend
  console.log(`${magenta}Starting React + Vite Frontend...${reset}`);

  if (fs.existsSync(VITE_BIN)) {
    frontendProcess = spawn(process.execPath, [VITE_BIN], {
      cwd: FRONTEND_DIR,
      stdio: ['ignore', 'pipe', 'pipe']
    });
  } else {
    const npmCmd = process.platform === 'win32' ? 'npm.cmd' : 'npm';
    frontendProcess = spawn(npmCmd, ['run', 'dev'], {
      cwd: FRONTEND_DIR,
      stdio: ['ignore', 'pipe', 'pipe']
    });
  }

  frontendProcess.stdout.on('data', (d) => {
    const msg = d.toString();
    msg.split(/\r?\n/).forEach((line) => {
      if (line.trim()) console.log(`${magenta}[frontend]${reset} ${line}`);
    });
  });

  frontendProcess.stderr.on('data', (d) => {
    const msg = d.toString();
    msg.split(/\r?\n/).forEach((line) => {
      if (line.trim()) console.log(`${magenta}[frontend]${reset} ${line}`);
    });
  });

  frontendProcess.on('exit', (code) => {
    if (!isShuttingDown) {
      console.log(`${red}[frontend] Process exited with code ${code}${reset}`);
      cleanShutdown();
    }
  });

  // 5. Final summary & launch browser
  setTimeout(() => {
    if (!isShuttingDown) {
      console.log(`\n${green}${bold}============================================================${reset}`);
      console.log(`${green}${bold}   COGNIVISION AI IS FULLY OPERATIONAL!                     ${reset}`);
      console.log(`${green}${bold}============================================================${reset}`);
      console.log(`   ➜  ${bold}Web Dashboard${reset} : ${cyan}${bold}http://localhost:5173/${reset}  ${yellow}(Opening in browser...)${reset}`);
      console.log(`   ➜  ${bold}FastAPI Backend${reset}: ${cyan}http://127.0.0.1:8001/${reset}`);
      console.log(`   ➜  ${bold}API Docs (UI)${reset}  : ${cyan}http://127.0.0.1:8001/docs${reset}`);
      console.log(`   ➜  ${bold}Health Status${reset}  : ${cyan}http://127.0.0.1:8001/api/health${reset}`);
      console.log(`${gray}   Press Ctrl+C anytime to stop both servers.${reset}\n`);

      openBrowser('http://localhost:5173/');
    }
  }, 2000);
}

main().catch((err) => {
  console.error(`${red}[Error] ${err.message}${reset}`);
  cleanShutdown();
});
