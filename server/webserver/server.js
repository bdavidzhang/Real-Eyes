import express from 'express';
import https from 'https';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

const distPath = path.join(__dirname, 'dist');

// Serve static files from dist (CSS, JS, assets)
app.use(express.static(distPath));

// HTML pages from the MPA build
const pages = ['plan', 'viewer', 'sender', 'detection-debug', 'summary'];

for (const page of pages) {
  app.get(`/${page}`, (req, res) => {
    res.sendFile(path.join(distPath, `${page}.html`));
  });
  app.get(`/${page}.html`, (req, res) => {
    res.sendFile(path.join(distPath, `${page}.html`));
  });
}

// Root serves index.html (landing page)
app.get('/', (req, res) => {
  res.sendFile(path.join(distPath, 'index.html'));
});

// HTTPS server
const httpsOptions = {
  key: fs.readFileSync(path.join(__dirname, 'server.key')),
  cert: fs.readFileSync(path.join(__dirname, 'server.cert')),
};

https.createServer(httpsOptions, app).listen(PORT, '0.0.0.0', () => {
  console.log(`Server running at https://0.0.0.0:${PORT}`);
});
