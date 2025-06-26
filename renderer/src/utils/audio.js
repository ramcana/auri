let mediaRecorder;
let audioChunks = [];

// Audio player for TTS responses
let audioPlayer = null;

// Audio queue system
let audioQueue = [];

// Audio state tracking
let audioPlaybackState = {
  isPlaying: false,
  currentAudioId: null,
  currentPartId: null,
};

// --- Audio Unlock ---
let isAudioUnlocked = false;

export const unlockAudio = () => {
  if (isAudioUnlocked) return;
  try {
    const context = new (window.AudioContext || window.webkitAudioContext)();
    if (context.state === 'suspended') {
      context.resume();
    }
    const buffer = context.createBuffer(1, 1, 22050);
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    source.start(0);
    if (context.state === 'suspended') {
      context.resume();
    }
    isAudioUnlocked = true;
    console.log('Audio context unlocked.');
  } catch (e) {
    console.error('Failed to unlock audio context:', e);
  }
};

export const checkMicrophoneAvailability = async () => {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const audioInputs = devices.filter(device => device.kind === 'audioinput');
    return audioInputs.length > 0;
  } catch (error) {
    console.error('Error checking microphone availability:', error);
    return false;
  }
};

export const startRecording = async () => {
  try {
    const hasMicrophone = await checkMicrophoneAvailability();
    if (!hasMicrophone) {
      throw new Error('No microphone detected on this device');
    }
    
    const constraints = {
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        sampleRate: 16000,
        channelCount: 1
      }
    };
    
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    unlockAudio();
    
    const mimeTypes = ['audio/webm', 'audio/wav', 'audio/ogg', 'audio/mp4', 'audio/webm;codecs=opus', 'audio/webm;codecs=pcm'];
    let selectedMimeType = '';
    for (const mimeType of mimeTypes) {
      if (MediaRecorder.isTypeSupported(mimeType)) {
        selectedMimeType = mimeType;
        break;
      }
    }
    
    const options = selectedMimeType ? { mimeType: selectedMimeType } : {};
    mediaRecorder = new MediaRecorder(stream, options);
    
    audioChunks = [];
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };
    
    mediaRecorder.start(100);
    console.log('Recording started');
  } catch (error) {
    console.error('Error accessing microphone:', error);
    if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
      throw new Error('Microphone not found.');
    } else if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
      throw new Error('Microphone access denied.');
    } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
      throw new Error('Microphone is already in use.');
    } else {
      throw error;
    }
  }
};

export const stopRecording = () => {
  return new Promise((resolve, reject) => {
    if (!mediaRecorder || mediaRecorder.state !== 'recording') {
      resolve(null);
      return;
    }
    
    mediaRecorder.onstop = () => {
      try {
        if (audioChunks.length === 0) {
          resolve(null);
          return;
        }
        
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        audioChunks = [];
        
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        
        resolve(audioBlob);
      } catch (error) {
        reject(error);
      }
    };
    
    mediaRecorder.stop();
  });
};

// This is a private function for the module
const playAudio = (base64Audio, format, onEndedCallback) => {
  try {
    if (!base64Audio) throw new Error('Invalid audio data');

    let cleanBase64 = base64Audio.includes(',') ? base64Audio.split(',')[1] : base64Audio;
    const binaryString = window.atob(cleanBase64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }

    const blob = new Blob([bytes], { type: `audio/${format}` });
    const audioUrl = URL.createObjectURL(blob);

    audioPlayer = new Audio(audioUrl);

    const cleanupAndCallback = () => {
      URL.revokeObjectURL(audioUrl);
      if (onEndedCallback) onEndedCallback();
    };

    audioPlayer.onended = cleanupAndCallback;
    audioPlayer.onerror = (e) => {
      console.error('Audio playback error:', e);
      cleanupAndCallback();
    };

        audioPlayer.play().catch(e => {
      console.error('Audio play() failed:', e);
      cleanupAndCallback();
    });

  } catch (error) {
    console.error('Error in playAudio:', error);
    if (onEndedCallback) onEndedCallback();
  }
};

const processAudioQueue = () => {
  if (audioPlaybackState.isPlaying || audioQueue.length === 0) {
    return;
  }

    const audioInfo = audioQueue.shift();
  
  audioPlaybackState.isPlaying = true;
  audioPlaybackState.currentAudioId = audioInfo.audioId;
  audioPlaybackState.currentPartId = audioInfo.partId;

  playAudio(audioInfo.base64Audio, audioInfo.format, () => {
    audioPlaybackState.isPlaying = false;
    audioPlaybackState.currentAudioId = null;
    audioPlaybackState.currentPartId = null;
    processAudioQueue();
  });
};

export const addToAudioQueue = (base64Audio, format = 'mp3', partId = 0, audioId = null) => {
  audioQueue.push({ base64Audio, format, partId, audioId });
  processAudioQueue();
};

export const stopAudioPlayback = () => {
  if (audioPlayer) {
    audioPlayer.pause();
    audioPlayer.removeAttribute('src');
    audioPlayer.load();
    audioPlayer = null;
  }
  audioPlaybackState.isPlaying = false;
  audioPlaybackState.currentAudioId = null;
  audioPlaybackState.currentPartId = null;
};

export const clearAudioQueue = () => {
  const queueLength = audioQueue.length;
  audioQueue = [];
  return queueLength;
};

export const getAudioPlaybackState = () => {
  return { ...audioPlaybackState };
};

export const getAudioQueue = () => {
  return [...audioQueue];
};
