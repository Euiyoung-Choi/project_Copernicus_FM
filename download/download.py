import ee
import geemap

# 1. Google 계정 인증 (최초 1회 필수)
# 실행하면 링크가 뜨거나 브라우저가 열립니다. 로그인을 완료하고 토큰을 입력하세요.
ee.Authenticate()
# 2. 인증 완료 후 프로젝트 ID를 넣어서 초기화
# 입력하신 프로젝트 ID 'aerobic-gift-464705-u5'를 사용합니다.
ee.Initialize(project='aerobic-gift-464705-u5')

# 3. ROI 설정 및 이후 코드 진행
roi = ee.Geometry.Rectangle([126.5, 34.8, 127.2, 35.3])
print("Earth Engine 초기화 성공!")