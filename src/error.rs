use thiserror::Error;

/// FC-ES 库错误类型。
#[derive(Error, Debug)]
pub enum FcesError {
    /// USearch 索引操作失败。
    #[error("USearch 错误: {0}")]
    UsSearch(String),

    /// 文件读写错误。
    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),

    /// InfoMap CLI 不可用。
    #[error("未找到 Infomap，请将其放入 PATH 或当前工作目录")]
    InfomapNotFound,

    /// InfoMap 执行失败。
    #[error("InfoMap 执行失败: {0}")]
    InfomapExecution(String),

    /// InfoMap 输出解析错误。
    #[error("InfoMap 解析错误: {0}")]
    InfomapParse(String),

    /// 输入数据无效。
    #[error("无效输入: {0}")]
    InvalidInput(String),
}
