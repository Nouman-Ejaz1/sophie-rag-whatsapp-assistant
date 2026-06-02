const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const http = require('http');

let FASTAPI_URL = 'http://127.0.0.1:8000/api/whatsapp/message';

// State trackers
let backendErrorNotified = false;
let whatsappReady = false;
let whatsappAuthenticated = false;
let whatsappInitStartedAt = null;
let whatsappWatchdog = null;
let lastQrAt = null;
let lastLoadingState = null;
const activeStatusMessages = new Map();

const LOCAL_DATA_DIR = path.resolve(__dirname, 'local_data');
const AUTH_DATA_DIR = path.join(LOCAL_DATA_DIR, '.wwebjs_auth');
const SESSION_DIR = path.join(AUTH_DATA_DIR, 'session');
const GATEWAY_LOCK_FILE = path.join(LOCAL_DATA_DIR, 'whatsapp_gateway.pid');
let WHATSAPP_HEADLESS = true;
let WHATSAPP_DEVTOOLS = false;
let DEBUG_PORT = 0;

// Robust, persistent local file logger that writes to local_data/gateway.log
function gatewayLog(level, message, error = null) {
    const timestamp = new Date().toISOString();
    let errStr = '';
    if (error) {
        errStr = ` | Error: ${error.message}`;
        if (error.stack) {
            errStr += `\nStack trace:\n${error.stack}\n`;
        }
    }
    const logLine = `[${timestamp}] [${level}] ${message}${errStr}\n`;
    
    // Output to stdout/stderr
    if (level === 'ERROR' || level === 'WARNING') {
        console.error(`⚠️ [${level}] ${message}`, error ? error.message : '');
    } else {
        console.log(`🤖 [${level}] ${message}`);
    }
    
    // Append to file
    try {
        const logDir = LOCAL_DATA_DIR;
        if (!fs.existsSync(logDir)) {
            fs.mkdirSync(logDir, { recursive: true });
        }
        fs.appendFileSync(path.join(logDir, 'gateway.log'), logLine, 'utf-8');
    } catch (e) {
        console.error('⚠️ Failed to append to local gateway log file:', e.message);
    }
}

gatewayLog('INFO', 'Initializing Sophie WhatsApp Gateway...');

// Simple manual .env parser to avoid requiring external dependencies
function loadEnv() {
    try {
        const envPath = path.resolve(__dirname, '.env');
        if (fs.existsSync(envPath)) {
            const content = fs.readFileSync(envPath, 'utf-8');
            content.split('\n').forEach(line => {
                const trimmed = line.trim();
                if (!trimmed || trimmed.startsWith('#')) return;
                const parts = trimmed.split('=');
                if (parts.length >= 2) {
                    const key = parts[0].trim();
                    const val = parts.slice(1).join('=').trim().replace(/^['"]|['"]$/g, '');
                    process.env[key] = val;
                }
            });
            gatewayLog('INFO', 'Loaded .env variables manually.');
        } else {
            gatewayLog('WARNING', 'No .env file found at backend/ root.');
        }
    } catch (e) {
        gatewayLog('ERROR', 'Failed to load .env file manually', e);
    }
}
loadEnv();
FASTAPI_URL = process.env.FASTAPI_URL || FASTAPI_URL;
WHATSAPP_HEADLESS = String(process.env.WHATSAPP_HEADLESS || 'true').toLowerCase() !== 'false';
WHATSAPP_DEVTOOLS = String(process.env.WHATSAPP_DEVTOOLS || 'false').toLowerCase() === 'true';
DEBUG_PORT = Number(process.env.WHATSAPP_DEBUG_PORT || '0');

function processExists(pid) {
    if (!pid || Number(pid) === process.pid) return false;
    try {
        process.kill(Number(pid), 0);
        return true;
    } catch (_) {
        return false;
    }
}

function acquireGatewayLock() {
    try {
        if (!fs.existsSync(LOCAL_DATA_DIR)) {
            fs.mkdirSync(LOCAL_DATA_DIR, { recursive: true });
        }
        if (fs.existsSync(GATEWAY_LOCK_FILE)) {
            const existingPid = fs.readFileSync(GATEWAY_LOCK_FILE, 'utf-8').trim();
            if (processExists(existingPid)) {
                gatewayLog(
                    'ERROR',
                    `Another WhatsApp gateway process is already running with PID ${existingPid}. ` +
                    `Stop it first, then run npm start again.`
                );
                process.exit(1);
            }
            gatewayLog('WARNING', `Removing stale gateway PID lock from old process ${existingPid || 'unknown'}.`);
        }
        fs.writeFileSync(GATEWAY_LOCK_FILE, String(process.pid), 'utf-8');
        gatewayLog('INFO', `Gateway process lock acquired. pid=${process.pid}`);
    } catch (e) {
        gatewayLog('ERROR', 'Failed to acquire WhatsApp gateway process lock', e);
        process.exit(1);
    }
}

function releaseGatewayLock() {
    try {
        if (fs.existsSync(GATEWAY_LOCK_FILE)) {
            const existingPid = fs.readFileSync(GATEWAY_LOCK_FILE, 'utf-8').trim();
            if (existingPid === String(process.pid)) {
                fs.unlinkSync(GATEWAY_LOCK_FILE);
            }
        }
    } catch (_) {}
}

acquireGatewayLock();

// Parse Whitelisted Allowed Numbers
const allowedNumbersRaw = process.env.ALLOWED_NUMBERS || '';
const allowedNumbers = allowedNumbersRaw.split(',').map(n => n.trim()).filter(n => n.length > 0);
if (allowedNumbers.length > 0) {
    gatewayLog('INFO', `Whitelisted WhatsApp Senders enabled: ${JSON.stringify(allowedNumbers)}`);
} else {
    gatewayLog('INFO', `Whitelisted Senders disabled. Replying to all incoming messages.`);
}

// Initialize WhatsApp client with local auth inside gitignored local_data directory
gatewayLog(
    'INFO',
    `WhatsApp auth path=${AUTH_DATA_DIR} | session path=${SESSION_DIR} | ` +
    `headless=${WHATSAPP_HEADLESS} | devtools=${WHATSAPP_DEVTOOLS} | debug_port=${DEBUG_PORT || 'auto'}`
);

const client = new Client({
    authStrategy: new LocalAuth({
        dataPath: AUTH_DATA_DIR,
        rmMaxRetries: 5
    }),
    takeoverOnConflict: true,
    takeoverTimeoutMs: 0,
    puppeteer: {
        headless: WHATSAPP_HEADLESS,
        devtools: WHATSAPP_DEVTOOLS,
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-extensions',
            '--disable-gpu',
            '--disable-features=site-per-process',
            '--window-size=1280,720',
            `--remote-debugging-port=${DEBUG_PORT || 0}`
        ]
    }
});

async function getWhatsAppDiagnostics() {
    const diag = {
        page_url: 'unavailable',
        page_title: 'unavailable',
        app_state: 'unavailable',
        body_hint: 'unavailable'
    };
    try {
        if (!client.pupPage) {
            return diag;
        }
        diag.page_url = client.pupPage.url();
        diag.page_title = await client.pupPage.title();
        const pageState = await client.pupPage.evaluate(() => {
            const bodyText = document.body ? document.body.innerText : '';
            return {
                appState:
                    window.AuthStore?.AppState?.state ||
                    window.Store?.AppState?.state ||
                    window.Debug?.VERSION ||
                    'unknown',
                bodyHint: bodyText ? bodyText.slice(0, 180).replace(/\s+/g, ' ') : ''
            };
        });
        diag.app_state = pageState.appState || 'unknown';
        diag.body_hint = pageState.bodyHint || '';
    } catch (e) {
        diag.error = e.message;
    }
    return diag;
}

function startWhatsAppWatchdog() {
    whatsappInitStartedAt = Date.now();
    if (whatsappWatchdog) {
        clearInterval(whatsappWatchdog);
    }
    whatsappWatchdog = setInterval(async () => {
        if (whatsappReady) {
            clearInterval(whatsappWatchdog);
            whatsappWatchdog = null;
            return;
        }
        const elapsedSec = Math.round((Date.now() - whatsappInitStartedAt) / 1000);
        const authState = whatsappAuthenticated ? 'authenticated' : 'not authenticated yet';
        const diag = await getWhatsAppDiagnostics();
        gatewayLog(
            elapsedSec >= 120 ? 'WARNING' : 'INFO',
            `WhatsApp client still initializing after ${elapsedSec}s (${authState}; last_loading=${lastLoadingState || 'none'}; ` +
            `last_qr=${lastQrAt ? new Date(lastQrAt).toISOString() : 'none'}; ` +
            `page=${diag.page_url}; title=${diag.page_title}; app_state=${diag.app_state}; hint=${diag.body_hint || diag.error || 'none'}). ` +
            `If this stays stuck, stop old gateway/chrome-for-testing processes. If QR never appears, run with WHATSAPP_HEADLESS=false or reset local_data/.wwebjs_auth.`
        );
    }, 30000);
}

// Generate QR Code in terminal for scanning
client.on('qr', (qr) => {
    lastQrAt = Date.now();
    whatsappAuthenticated = false;
    gatewayLog('INFO', 'New QR Code generated. Scan with WhatsApp Link Device options.');
    qrcode.generate(qr, { small: true });
});

// Client Ready Event
client.on('ready', () => {
    whatsappReady = true;
    whatsappAuthenticated = true;
    if (whatsappWatchdog) {
        clearInterval(whatsappWatchdog);
        whatsappWatchdog = null;
    }
    gatewayLog('INFO', 'Sophie WhatsApp Client is READY and CONNECTED!');
    console.log('💬 Send a text message to link numbers. Deleting backend/local_data/.wwebjs_auth resets numbers!');
});

// Authenticated Event
client.on('authenticated', () => {
    whatsappAuthenticated = true;
    gatewayLog('INFO', 'Session successfully AUTHENTICATED with WhatsApp!');
});

// Loading Screen Event
client.on('loading_screen', (percent, message) => {
    lastLoadingState = `${percent}% ${message || ''}`.trim();
    gatewayLog('INFO', `Loading WhatsApp Web: ${percent}% - ${message}`);
});

// Auth Failure Event
client.on('auth_failure', (msg) => {
    whatsappAuthenticated = false;
    gatewayLog(
        'ERROR',
        `AUTHENTICATION FAILURE: ${msg}. The saved session may be invalid. ` +
        `Stop the gateway and rename/delete local_data/.wwebjs_auth to force a fresh QR login.`
    );
});

// Disconnected Event
client.on('disconnected', (reason) => {
    whatsappReady = false;
    whatsappAuthenticated = false;
    gatewayLog('WARNING', `Sophie WhatsApp Client was DISCONNECTED: ${reason}`);
});

// State Change Event
client.on('change_state', (state) => {
    gatewayLog('INFO', `Sophie WhatsApp Client state changed to: ${state}`);
});

// Helper to sanitize output, extracting only user_output if present, or stripping XML tags as safety fallback
function cleanOutput(reply) {
    if (!reply) return '';
    // 1. Try to extract user_output tag content
    const match = reply.match(/<user_output>([\s\S]*?)<\/user_output>/i);
    if (match && match[1]) {
        return match[1].trim();
    }
    // 2. If no user_output tag found, strip all <ouput>, <tools_call>, <thinking> etc. tags to preserve clean text
    let cleaned = reply;
    cleaned = cleaned.replace(/<ouput>([\s\S]*?)<\/ouput>/gi, '$1');
    cleaned = cleaned.replace(/<tools_call>([\s\S]*?)<\/tools_call>/gi, '');
    cleaned = cleaned.replace(/<thinking>([\s\S]*?)<\/thinking>/gi, '');
    cleaned = cleaned.replace(/<\/?(ouput|user_output|tools_call|thinking)>/gi, '');
    return cleaned.trim();
}

// Handle incoming messages and self-chats
client.on('message_create', async (msg) => {
    const sender = msg.from;
    const cleanSender = sender.split('@')[0];
    const text = msg.body || '';
    
    // Raw telemetry console print to capture ALL events
    console.log(`📡 [RAW EVENT] message_create fired | from: ${sender} | to: ${msg.to} | fromMe: ${msg.fromMe} | text: "${text.substring(0, 40)}"`);

    // Only respond to private chats (ignore group chats)
    if (sender.endsWith('@c.us') || sender.endsWith('@lid')) {

        // Determine if this is an incoming message or a self-chat message
        const isSelfChat = msg.fromMe && (msg.to === msg.from);
        
        // Guard rails: Ignore standard outbound messages sent by us to other contacts
        if (msg.fromMe && !isSelfChat) return;
        
        // Guard rails: Ignore our own automated responses in self-chat to avoid loops
        if (msg.fromMe && isSelfChat && (text.startsWith('<ouput>') || text.startsWith('⚠️') || text.startsWith('[INFO]'))) return;

        // Whitelist validation check
        if (allowedNumbers.length > 0 && !allowedNumbers.includes('*')) {
            const isAuthorized = allowedNumbers.includes(sender) || allowedNumbers.includes(cleanSender);
            if (!isAuthorized) {
                // If it is a self-chat by the logged-in owner, bypass whitelist check
                if (!isSelfChat) {
                    gatewayLog('WARNING', `Ignored unauthorized message from sender: ${sender}`);
                    return;
                }
            }
        }

        gatewayLog('INFO', `Received message from ${sender} (Self-Chat: ${isSelfChat}): "${text}"`);

        // Start WhatsApp typing/thinking animation indicator immediately and keep it active
        let chat;
        let typingInterval = null;
        try {
            chat = await msg.getChat();
            await chat.sendStateTyping();
            typingInterval = setInterval(async () => {
                try {
                    await chat.sendStateTyping();
                } catch (err) {
                    gatewayLog('WARNING', `Failed to sendStateTyping in heartbeat interval`, err);
                }
            }, 10000); // 10-second heartbeat
        } catch (err) {
            gatewayLog('WARNING', `Failed to trigger sendStateTyping indicator`, err);
        }

        // Send initial dynamic status message
        let statusMsg = null;
        try {
            statusMsg = await msg.reply("🤖 *Sophie is thinking...*");
            activeStatusMessages.set(sender, statusMsg);
        } catch (err) {
            gatewayLog('WARNING', `Failed to send initial status message placeholder`, err);
        }

        try {
            let mediaPayload = {};
            if (msg.hasMedia) {
                try {
                    const media = await msg.downloadMedia();
                    if (media && media.data) {
                        mediaPayload = {
                            media_type: msg.type || 'media',
                            mime_type: media.mimetype || '',
                            filename: media.filename || msg._data?.filename || '',
                            media_base64: media.data
                        };
                        gatewayLog('INFO', `Downloaded WhatsApp media for backend: type=${mediaPayload.media_type}, mime=${mediaPayload.mime_type}, filename=${mediaPayload.filename || 'none'}`);
                    }
                } catch (mediaErr) {
                    gatewayLog('WARNING', 'Failed to download WhatsApp media payload', mediaErr);
                }
            }

            // Forward the message to our FastAPI Backend
            const backendPayload = {
                sender: sender,
                message: text,
                ...mediaPayload
            };
            gatewayLog(
                'INFO',
                `Posting to FastAPI ${FASTAPI_URL} | sender=${sender} | text_len=${text.length} | media=${Object.keys(mediaPayload).length > 0 ? mediaPayload.media_type : 'none'}`
            );
            const started = Date.now();
            const response = await axios.post(FASTAPI_URL, backendPayload);
            gatewayLog(
                'INFO',
                `FastAPI response ${response.status} in ${Date.now() - started}ms | app_status=${response.data?.status || 'unknown'} | reply_len=${(response.data?.response || '').length}`
            );

            let reply = response.data.response || '';
            
            // Clean XML tags to send ONLY the conversational user_output text to WhatsApp
            reply = cleanOutput(reply);
            
            // Connection succeeded. Reset error indicator
            backendErrorNotified = false;

            // Reply directly to the user on WhatsApp by editing the status message
            if (statusMsg) {
                try {
                    await statusMsg.edit(reply);
                    gatewayLog('INFO', `Edited status message successfully for ${sender}`);
                } catch (editErr) {
                    gatewayLog('WARNING', `Failed to edit status message, falling back to new reply`, editErr);
                    await msg.reply(reply);
                }
                activeStatusMessages.delete(sender);
            } else {
                await msg.reply(reply);
                gatewayLog('INFO', `Replied successfully to ${sender} (no status message found)`);
            }
        } catch (error) {
            gatewayLog('ERROR', `Error communicating with FastAPI backend at ${FASTAPI_URL}`, error);
            if (error.response) {
                gatewayLog(
                    'ERROR',
                    `FastAPI returned ${error.response.status}: ${JSON.stringify(error.response.data).substring(0, 500)}`
                );
            }
            
            const errMsg = '⚠️ Error connecting to Sophie\'s reasoning core. Is your backend server online?';
            if (statusMsg) {
                try {
                    await statusMsg.edit(errMsg);
                } catch (editErr) {
                    await msg.reply(errMsg);
                }
                activeStatusMessages.delete(sender);
            } else if (!backendErrorNotified) {
                await msg.reply(errMsg);
                backendErrorNotified = true;
            }
        } finally {
            activeStatusMessages.delete(sender);
            if (typingInterval) {
                clearInterval(typingInterval);
            }
            try {
                if (chat) {
                    await chat.clearState();
                }
            } catch (err) {
                gatewayLog('WARNING', `Failed to clear typing state`, err);
            }
        }
    }
});

// Lightweight HTTP Server for Outbound Messages on Port 3001
const server = http.createServer(async (req, res) => {
    if (req.method === 'GET' && req.url === '/health') {
        const initializingSeconds = whatsappInitStartedAt && !whatsappReady
            ? Math.round((Date.now() - whatsappInitStartedAt) / 1000)
            : 0;
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({
            status: whatsappReady ? 'ready' : 'initializing',
            ready: whatsappReady,
            authenticated: whatsappAuthenticated,
            initializing_seconds: initializingSeconds,
            fastapi_url: FASTAPI_URL,
            pid: process.pid,
            auth_path: AUTH_DATA_DIR,
            headless: WHATSAPP_HEADLESS,
            last_loading_state: lastLoadingState,
            last_qr_at: lastQrAt ? new Date(lastQrAt).toISOString() : null,
            whatsapp_web_js_version: require('whatsapp-web.js/package.json').version
        }));
    }

    if (req.method === 'POST' && req.url === '/status') {
        let body = '';
        req.on('data', chunk => { body += chunk; });
        req.on('end', async () => {
            try {
                const data = JSON.parse(body);
                if (!data.sender || !data.text) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    return res.end(JSON.stringify({ error: 'Missing sender or text field' }));
                }
                const recipient = data.sender;
                const text = data.text;
                
                gatewayLog('INFO', `Received status update for ${recipient}: "${text}"`);
                const statusMsg = activeStatusMessages.get(recipient);
                if (statusMsg) {
                    await statusMsg.edit(text);
                } else {
                    gatewayLog('WARNING', `No active status message found for ${recipient}`);
                }
                
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ status: 'success' }));
            } catch (err) {
                gatewayLog('ERROR', `Failed to process status update`, err);
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: err.message }));
            }
        });
    }

    else if (req.method === 'POST' && req.url === '/send') {
        let body = '';
        req.on('data', chunk => { body += chunk; });
        req.on('end', async () => {
            let recipient = 'unknown';
            try {
                const data = JSON.parse(body);
                if (!data.to || !data.message) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    return res.end(JSON.stringify({ error: 'Missing to or message field' }));
                }
                
                recipient = data.to;
                // Append @c.us if missing
                if (!recipient.endsWith('@c.us') && !recipient.endsWith('@g.us')) {
                    recipient = `${recipient}@c.us`;
                }
                
                gatewayLog('INFO', `Sending outbound scheduled alert to ${recipient}: "${data.message.substring(0, 50)}..."`);
                await client.sendMessage(recipient, data.message);
                
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ status: 'success' }));
            } catch (err) {
                gatewayLog('ERROR', `Failed to send outbound message to ${recipient}`, err);
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: err.message }));
            }
        });
    } else {
        res.writeHead(404);
        res.end();
    }
});

server.listen(3001, () => {
    gatewayLog('INFO', 'Outbound message server listening on port 3001');
});

server.on('error', (error) => {
    gatewayLog('ERROR', 'Outbound message server failed. Port 3001 may already be in use.', error);
    releaseGatewayLock();
    process.exit(1);
});

function shutdownGateway(signal) {
    gatewayLog('INFO', `Received ${signal}. Closing WhatsApp gateway cleanly...`);
    releaseGatewayLock();
    process.exit(0);
}

process.on('SIGINT', () => shutdownGateway('SIGINT'));
process.on('SIGTERM', () => shutdownGateway('SIGTERM'));
process.on('exit', releaseGatewayLock);

process.on('unhandledRejection', (reason) => {
    gatewayLog('ERROR', 'Unhandled promise rejection in WhatsApp gateway', reason instanceof Error ? reason : new Error(String(reason)));
});

process.on('uncaughtException', (error) => {
    gatewayLog('ERROR', 'Uncaught exception in WhatsApp gateway', error);
});

gatewayLog('INFO', `Starting WhatsApp Web client initialization. FastAPI URL=${FASTAPI_URL}`);
startWhatsAppWatchdog();
client.initialize()
    .then(() => {
        gatewayLog('INFO', 'client.initialize() promise resolved; waiting for authenticated/ready events if not already fired.');
    })
    .catch((error) => {
        gatewayLog('ERROR', 'client.initialize() failed', error);
    });
