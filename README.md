# warp_api
画像のbase64形式のデータをjsonで受け取ると補正した画像データのbase64形式を返すエンドポイントを持つAPI  

# 用途
Railsのサービスに使うためFlaskで作成  
ポート：5000   

# request body  
json: {
  image: {
    before_base: ...,
    clicked_position: [x1, x2],
    paper_size : [pos1, pos2],
  },
}  

# response body
json: {
  status: 200,
  data: {
    after_image: ...,
  },
}  


