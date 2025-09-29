// Programa: Ponte H L298N com “giro leve” por comando
// Motor de passo controlado via comandos seriais 'R' (direita)

const int IN1 = 8;
const int IN2 = 9;
const int IN3 = 10;
const int IN4 = 11;

// Delay entre cada passo (ms)
const int tempo = 10;

// Quantos passos dar a cada comando
const int STEPS_PER_CLICK = 5;

// Guarda em que fase de 0 a 3 estamos
int currentStep = 0;

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  Serial.begin(9600);
  Serial.println("Envie 'R' para girar poucos passos.");
}

// Executa um passo na sequência 0→1→2→3
void passo(int seq) {
  switch(seq) {
    case 0:
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);
      break;
    case 1:
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN3, LOW);
      digitalWrite(IN4, HIGH);
      break;
    case 2:
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, HIGH);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
      break;
    case 3:
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      digitalWrite(IN3, HIGH);
      digitalWrite(IN4, LOW);
      break;
  }
  delay(tempo);
}

// Desliga as bobinas do motor (evita torque residual)
void desligarMotor() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void loop() {
  if (Serial.available() > 0) {
    char c = Serial.read();

    if (c == 'R' || c == 'r') {
      for (int i = 0; i < STEPS_PER_CLICK; i++) {
        passo(currentStep);
        currentStep = (currentStep + 1) % 4;
      }

      desligarMotor();

      Serial.println("Pronto");

      while (Serial.available() > 0) {
        Serial.read();
      }
    }
  }
}
