require('dotenv').config();
const { AssemblyAI } = require('assemblyai');
const WebSocket = require('ws');

// Check for API Key
const apiKey = process.env.ASSEMBLYAI_API_KEY;
if (!apiKey) {
  console.error('FATAL ERROR: ASSEMBLYAI_API_KEY is not set. Please create a .env file in the root directory with your API key.');
  process.exit(1); // Exit with an error code
}

try {
  const client = new AssemblyAI({ apiKey });
  const wss = new WebSocket.Server({ port: 8080 });

  wss.on('listening', () => {
    console.log('WebSocket server started and listening on port 8080');
  });

  wss.on('connection', ws => {
    console.log('Client connected');

    ws.on('message', async (message) => {
      try {
        const transcript = await client.transcripts.create({
          audio: message,
        });

        if (transcript.text) {
          console.log('Transcription:', transcript.text);
          ws.send(JSON.stringify({ type: 'transcription_result', text: transcript.text }));
        } else {
          console.log('Transcription failed.');
          ws.send(JSON.stringify({ type: 'transcription_error', text: 'Sorry, I could not understand that.' }));
        }
      } catch (error) {
        console.error('Error during transcription:', error);
        ws.send(JSON.stringify({ type: 'transcription_error', text: 'An error occurred during transcription.' }));
      }
    });

    ws.on('close', () => {
      console.log('Client disconnected');
    });
  });

  wss.on('error', (error) => {
    console.error('WebSocket server error:', error);
    process.exit(1);
  });

} catch (error) {
  console.error('Failed to initialize server:', error);
  process.exit(1);
}
