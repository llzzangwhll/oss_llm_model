import argparse
from transformers import pipeline


# IDE에서 디버깅할 때 사용할 기본 프롬프트
DEFAULT_PROMPT = "양자역학을 명확하고 간결하게 설명해주세요."


def main():
    parser = argparse.ArgumentParser(
        description="GPT-OSS-20B 텍스트 생성 스크립트"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"텍스트 생성을 위한 입력 프롬프트 (기본값: '{DEFAULT_PROMPT[:50]}...')"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="생성할 최대 토큰 수 (기본값: 256)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-20b",
        help="사용할 모델 ID (기본값: openai/gpt-oss-20b)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="샘플링 온도 (선택사항)"
    )
    args = parser.parse_args()

    # 파이프라인 초기화
    # 참고: openai/gpt-oss-20b는 이미 Mxfp4 양자화가 적용된 모델입니다
    print(f"모델 로딩 중: {args.model}...")
    print("(이 모델은 이미 Mxfp4 양자화가 적용되어 메모리 효율적입니다)")
    pipe = pipeline(
        "text-generation",
        model=args.model,
        torch_dtype="auto",
        device_map="auto",
    )

    # 메시지 준비
    messages = [
        {"role": "user", "content": args.prompt},
    ]

    # 텍스트 생성
    print(f"\n프롬프트: {args.prompt}")
    print("\n응답 생성 중...\n")
    generation_kwargs = {"max_new_tokens": args.max_tokens}
    if args.temperature is not None:
        generation_kwargs["temperature"] = args.temperature

    outputs = pipe(messages, **generation_kwargs)

    # 결과 출력
    print("=" * 80)
    print(outputs[0]["generated_text"][-1]["content"])
    print("=" * 80)


if __name__ == "__main__":
    main()
