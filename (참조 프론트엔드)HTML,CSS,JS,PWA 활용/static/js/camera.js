let mediaRecorder;
let audioStream, videoStream;
let chunks = [];

let audioContext, analyser, source, dataArray;
let silenceDuration = 0;
const SILENCE_THRESHOLD = 0.02;
const SILENCE_LIMIT = 2000;

function camInit(userStream) {
  const video = document.getElementById("cameraview");
  video.srcObject = userStream;
  video.play();
}

function drawFaceBoxes(boxes) {
  const video = document.getElementById("cameraview");
  const canvas = document.getElementById("overlay");
  const ctx = canvas.getContext("2d");

  const videoRect = video.getBoundingClientRect();
  canvas.style.position = "absolute";
  canvas.style.top = `${videoRect.top}px`;
  canvas.style.left = `${videoRect.left}px`;
  canvas.style.width = `${videoRect.width}px`;
  canvas.style.height = `${videoRect.height}px`;

  canvas.width = videoRect.width;
  canvas.height = videoRect.height;

  const scaleX = videoRect.width / 640;
  const scaleY = videoRect.height / 480;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 3;
  ctx.strokeStyle = "lime";

  boxes.forEach(box => {
    const { x, y, width, height } = box;
    ctx.strokeRect(x * scaleX, y * scaleY, width * scaleX, height * scaleY);
  });
}

function setupAudioMonitor() {
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  source = audioContext.createMediaStreamSource(audioStream);
  source.connect(analyser);
  dataArray = new Uint8Array(analyser.fftSize);
}

function monitorSilence() {
  analyser.getByteTimeDomainData(dataArray);
  const volume = getVolume(dataArray);

  console.log("볼륨:", volume.toFixed(5));

  if (volume < SILENCE_THRESHOLD) {
    silenceDuration += 100;
    if (silenceDuration >= SILENCE_LIMIT) {
      console.log("무음 감지 → 녹음 종료");
      if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
      }
      return;
    }
  } else {
    silenceDuration = 0;
  }
  setTimeout(monitorSilence, 100);
}

function getVolume(data) {
  let sum = 0;
  for (let i = 0; i < data.length; i++) {
    const val = (data[i] - 128) / 128;
    sum += val * val;
  }
  return Math.sqrt(sum / data.length);
}

function showSubtitle(text, duration = 4000) {
  const subtitle = document.getElementById("subtitle");
  if (!subtitle) return;
  subtitle.textContent = text;
  subtitle.classList.remove("subtitle-hidden");
  subtitle.classList.add("subtitle-visible");

  if (duration > 0) {
    setTimeout(() => {
      subtitle.classList.remove("subtitle-visible");
      subtitle.classList.add("subtitle-hidden");
    }, duration);
  }
}

function showCharacterState(state) {
  const body = document.getElementById("c_5");
  if (!body) return;
  body.classList.remove("listen", "normal");
  body.classList.add(state === "listen" ? "listen" : "normal");
}

function startConversation() {
  if (!audioStream) {
    console.error("오디오 스트림이 초기화되지 않았습니다.");
    return;
  }

  chunks = [];
  setupAudioMonitor();

  try {
    mediaRecorder = new MediaRecorder(audioStream, { mimeType: 'audio/webm;codecs=opus' });
  } catch (err) {
    console.error("MediaRecorder 초기화 실패:", err);
    return setTimeout(startConversation, 1000);
  }

  mediaRecorder.ondataavailable = e => {
    console.log("ondataavailable 호출됨", e.data?.size);
    chunks.push(e.data);
  };

  mediaRecorder.onstop = () => {
    const blob = new Blob(chunks, { type: 'audio/webm' });
    console.log("녹음 종료, Blob 크기:", blob.size);

    if (blob.size === 0) {
      console.warn("빈 오디오 Blob, 전송 생략");
      return setTimeout(startConversation, 100);
    }

    showSubtitle("이야기를 해주세요! 열심히 듣고 있습니다!", 0);
    showCharacterState("listen");

    const form = new FormData();
    form.append("file", blob, "input.webm");

    fetch("https://nfredmpfjwwrknjr.tunnel.elice.io/child/converse", {
      method: "POST",
      body: form,
      credentials: "include",
    })
    .then(res => {
      if (!res.ok) throw new Error(`서버 응답 실패: ${res.status}`);
      return res.json();
    })
    .then(data => {
      if (!data.audio_url) {
        showSubtitle("(응답 없음)", 2000);
        showCharacterState("normal");
        return setTimeout(startConversation, 100);
      }

      const audio = new Audio("https://nfredmpfjwwrknjr.tunnel.elice.io" + data.audio_url + `?t=${Date.now()}`);
      if (data.text) {
        showSubtitle(data.text, 5000);
      } else {
        showSubtitle("(응답 없음)", 2000);
      }
      showCharacterState("normal");

      audio.onended = () => {
        showSubtitle("", 0);
        console.log("응답 재생 완료 → 다음 대화 시작");
        setTimeout(startConversation, 100);
      };

      audio.play().catch(err => {
        console.error("오디오 재생 실패:", err);
        showSubtitle("응답 재생 실패", 2000);
        setTimeout(startConversation, 1000);
      });
    })
    .catch(err => {
      console.error("오디오 전송 실패 또는 서버 에러:", err);
      showSubtitle("서버 오류, 다시 시도합니다.", 2000);
      setTimeout(startConversation, 1000);
    });
  };

  try {
    mediaRecorder.start(1000);  // 1초 타임슬라이스
    console.log("녹음 시작됨");
    silenceDuration = 0;
    setTimeout(monitorSilence, 500);  // 초기 지연 후 감시 시작
    showSubtitle("이야기를 해주세요! 열심히 듣고 있습니다!", 0);
    showCharacterState("listen");
  } catch (err) {
    console.error("녹음 시작 실패:", err);
    showSubtitle("녹음 실패, 다시 시도합니다.", 2000);
    setTimeout(startConversation, 1000);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(vs => {
      videoStream = vs;
      camInit(videoStream);
      setInterval(() => {
        if (videoStream) sendEmotionDetectionRequest();
      }, 5000);
    })
    .catch(err => {
      console.error("비디오 접근 실패:", err);
    });

  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(as => {
      audioStream = as;
      startConversation();
    })
    .catch(err => {
      console.error("오디오 접근 실패:", err);
      alert("마이크 권한이 필요합니다.");
    });
});

function captureFrame() {
  const video = document.getElementById("cameraview");
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext("2d");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  return new Promise(resolve => {
    canvas.toBlob(blob => resolve(blob), "image/jpeg");
  });
}

async function sendEmotionDetectionRequest() {
  try {
    const imageBlob = await captureFrame();
    const formData = new FormData();
    formData.append("file", imageBlob, "frame.jpg");

    fetch("https://nfredmpfjwwrknjr.tunnel.elice.io/child/emo_detect", {
      method: "POST",
      body: formData,
      credentials: "include",
    })
    .then(res => res.json())
    .then(result => {
      if (result.faces?.length > 0) {
        console.log("감정 분석:", result.faces.map(f => f.emotion).join(", "));
        const boxes = result.faces
          .filter(f => f.box && typeof f.box.x === "number")
          .map(f => f.box);
        drawFaceBoxes(boxes);
      }
    })
    .catch(err => {
      console.error("감정 분석 실패:", err);
    });
  } catch (err) {
    console.error("프레임 캡처 실패:", err);
  }
}
